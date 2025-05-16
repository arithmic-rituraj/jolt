use super::{
    bmmtv::Error, kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG}
};
use crate::{field::JoltField, poly::commitment::bmmtv::afgho::AfghoCommitment};
use crate::poly::multilinear_polynomial::{MultilinearPolynomial};
use crate::utils::transcript::Transcript;
use crate::{
    msm::{Icicle, VariableBaseMSM},
    poly::commitment::kzg::SRS,
    utils::transcript::AppendToTranscript,
};
use anyhow::Ok;
use ark_ff::PrimeField;
use ark_ec::{pairing::{Pairing, PairingOutput}, AffineRepr, CurveGroup, FixedBase};
use ark_std::{One, UniformRand};
use rand_core::{CryptoRng, RngCore};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{marker::PhantomData, sync::Arc};
use ark_ff::Field;

#[derive(Clone)]
pub struct TwoLevelSRS<P: Pairing>
where
    P::G1: Icicle,
{
    pub kzg_srs: SRS<P>,
    pub afgho_srs: Vec<P::G2Affine>,
}
#[derive(Clone)]
pub struct MercurySRS<P: Pairing>(Arc<TwoLevelSRS<P>>)
where
    P::G1: Icicle;

impl<P: Pairing> MercurySRS<P>
where
    P::G1: Icicle,
{
    pub fn setup<R: RngCore + CryptoRng>(mut rng: R, num_vars: usize) -> Self
    where
        P::ScalarField: JoltField,
    {
        // SRS constains KZG SRS corresponding to [1]_1, [\rho]_1 ..., [\rho^{m-1}]_1 and [1]_2, [\sigma]_2 ..., [\sigma^{m-1}]_2
        // where m = 2^{num_vars/2} and sigma = rho^m.
        // Compute it like KZG SRS is computed in SRS::setup.

        let m = 1 << ((num_vars as f64 / 2.0).ceil() as usize);

        // beta required for sigma = rho^m
        let beta = P::ScalarField::rand(&mut rng);

        let g1 = P::G1::rand(&mut rng);
        let g2 = P::G2::rand(&mut rng);

        // TOD: Change it to 2 because only 2 G2_POWERS are required for KZG
        // This is just for testing of AFGHO SRS
        let num_g2_powers = 256; //256 works for num_vars <=8

        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let g1_window_size = FixedBase::get_mul_window_size(m);
        let g2_window_size = FixedBase::get_mul_window_size(num_g2_powers);
        let g1_table = FixedBase::get_window_table(scalar_bits, g1_window_size, g1);
        let g2_table = FixedBase::get_window_table(scalar_bits, g2_window_size, g2);

        let (g1_powers_projective, g2_powers_projective) = rayon::join(
            || {
                let beta_powers: Vec<P::ScalarField> = (0..m)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();
                FixedBase::msm(
                        scalar_bits,
                        g1_window_size,
                        &g1_table,
                        &beta_powers,
                )
                    
            },
            || {
                let beta_powers: Vec<P::ScalarField> = (0..=num_g2_powers)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();

                FixedBase::msm(scalar_bits, g2_window_size, &g2_table, &beta_powers)
            },
        );

        let (g1_powers, g2_powers) = rayon::join(
            || P::G1::normalize_batch(&g1_powers_projective),
            || P::G2::normalize_batch(&g2_powers_projective),
        );

        // Precompute a commitment to each power-of-two length vector of ones, which is just the sum of each power-of-two length prefix of the SRS
        let num_powers = (g1_powers.len() as f64).log2().floor() as usize + 1;
        let all_ones_coeffs: Vec<u8> = vec![1; m + 1];
        let powers_of_2 = (0..num_powers).into_par_iter().map(|i| 1usize << i);
        let g_products = powers_of_2
            .map(|power| {
                <P::G1 as VariableBaseMSM>::msm_u8(
                    &g1_powers[..power],
                    &all_ones_coeffs[..power],
                    Some(1),
                )
                .unwrap()
                .into_affine()
            })
            .collect();

        let gpu_g1 = None;

        let kzg_srs = SRS {
            g1_powers,
            g2_powers: g2_powers.clone(),
            g_products,
            gpu_g1,
        };

        let afgho_srs_len = 1 << (num_vars - m.trailing_zeros() as usize);

        // sigma = rho^m
        let sigma = beta.pow([m as u64]);

        let g2_window_size = FixedBase::get_mul_window_size(afgho_srs_len);

        let g2_table = FixedBase::get_window_table(scalar_bits, g2_window_size, g2);

        let sigma_powers: Vec<P::ScalarField> = (0..afgho_srs_len - 1)
            .scan(sigma, |acc, _| {
                let val = *acc;
                *acc *= sigma;
                Some(val)
            })
            .collect();
        let g2_powers_projective_afgho =
        FixedBase::msm(scalar_bits, g2_window_size, &g2_table, &sigma_powers);
        let mut g2_powers_afgho: Vec<_> = P::G2::normalize_batch(&g2_powers_projective_afgho);
        g2_powers_afgho.insert(0, g2_powers[0]);

        Self(Arc::new(TwoLevelSRS {
            kzg_srs,
            afgho_srs: g2_powers_afgho.into_iter().map(|g| g.into()).collect(),
        }))
    }

    pub fn trim(self, max_degree: usize) -> (MercuryProverKey<P>, MercuryVerifierKey<P>) {
        // TODO: ARC change
        let m = (max_degree as f64 / 2.0).ceil() as usize; //dummy
        let (_, kzg_vk) = SRS::trim(Arc::new(self.0.kzg_srs.clone()), m);

        (
            MercuryProverKey {
                srs: self.clone(),
                num_vars: max_degree,
            },
            MercuryVerifierKey {
                kzg_vk,
                afgho_vk: self.0.afgho_srs[1],
            },
        )
    }
}

pub struct MercuryProverKey<P: Pairing>
where
    P::G1: Icicle,
{
    pub srs: MercurySRS<P>,
    pub num_vars: usize,
}

pub struct MercuryVerifierKey<P: Pairing>
where
    P::G1: Icicle,
{
    pub kzg_vk: KZGVerifierKey<P>,
    pub afgho_vk: P::G2Affine,
}

pub struct MercuryCommitment<P: Pairing>(pub P::TargetField);

impl<P: Pairing> Default for MercuryCommitment<P> {
    fn default() -> Self {
        Self(P::TargetField::one())
    }
}

impl<P: Pairing> AppendToTranscript for MercuryCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        // Append all 12 F_p elements in the commitment
        let fp12 = &self.0;
        // let c0 = fp12.

        // let c0 = &fp12.c0;

        // todo()!
    }
}

#[derive(Clone)]
pub struct Mercury<P: Pairing, ProofTranscript: Transcript> {
    _phantom: PhantomData<(P, ProofTranscript)>,
}

impl<P: Pairing, ProofTranscript: Transcript> Mercury<P, ProofTranscript>
    where <P as Pairing>::ScalarField: JoltField, <P as Pairing>::G1: Icicle
{
    pub fn protocol_name() -> &'static [u8] {
        b"Mercury"
    }

    pub fn commit(
        pp: &MercuryProverKey<P>,
        poly: &MultilinearPolynomial<P::ScalarField>,
    ) -> Result<PairingOutput<P>, Error>
    where
        P::ScalarField: JoltField,
    {
        // Implement like commit in poly_commit.rs.
        let num_vars = poly.len().trailing_zeros() as usize;
        let m = (num_vars as f64 / 2.0).ceil() as usize;

        let num_rows = 1 << m;
        let num_cols = 1 << (num_vars - m);
        assert_eq!(poly.len(), num_rows * num_cols);
        let kzg_pk = KZGProverKey::new(Arc::new(pp.srs.0.kzg_srs.clone()), 0, num_rows);

        // 1. Compute KZG commitments to each column
        let mut col_commitments = Vec::with_capacity(num_cols);
        for col in 0..num_cols {
            let mut column = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let idx = col * num_rows + row;
                column.push(poly.get_coeff(idx));
            }

            let com = UnivariateKZG::commit_as_univariate(&kzg_pk, &MultilinearPolynomial::from(column))?;

            col_commitments.push(com.into_group());
        }

        let mut ck = Vec::new();
        for i in 0..pp.srs.0.afgho_srs[..col_commitments.len()].len(){
            ck.push(pp.srs.0.afgho_srs[i].into_group())
        }


        // 2. AFGHO commitment to the vector of column commitments
        let afgho_commitment = AfghoCommitment::<P>::commit(
            &ck,
            &col_commitments,
        )?;

        Ok(afgho_commitment)
    }

}

#[cfg(test)]
mod tests{
    use ark_bn254::{Bn254, Fq12Config, Fr};
    use ark_ff::{CyclotomicMultSubgroup, FftField, Fp12ConfigWrapper, PrimeField, QuadExtField, UniformRand};
    use rand_core::SeedableRng;
    use crate::{poly::{commitment::{hyperkzg::{HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey}, mercury::{Mercury, MercuryProverKey, MercurySRS, MercuryVerifierKey}}, multilinear_polynomial::MultilinearPolynomial, unipoly::UniPoly}, utils::transcript::KeccakTranscript};

    #[test]
    fn test_mercury_commit(){
        let rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let srs = MercurySRS::setup(rng, 3);
        let (pk, _): (MercuryProverKey<Bn254>, MercuryVerifierKey<Bn254>) = srs.trim(2);

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let beta = Fr::rand(&mut rng);

        let uni_poly = UniPoly::from_coeff(vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let multi_poly = MultilinearPolynomial::from(vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let eval = uni_poly.evaluate(&beta);

        let C = Mercury::<_, KeccakTranscript>::commit(&pk, &multi_poly).unwrap();
        let commit: QuadExtField<Fp12ConfigWrapper<Fq12Config>> = C.0;
        let expected_commit  = QuadExtField::GENERATOR.cyclotomic_exp(eval.into_bigint());
        assert_eq!(expected_commit, commit)
    }

    #[test]
    fn check_setup(){
        let rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let num_vars = 6;
        let mercury_srs: MercurySRS<ark_ec::bn::Bn<ark_bn254::Config>> = MercurySRS::setup(rng, num_vars);
        let (mercury_pk, mercury_vk): (MercuryProverKey<Bn254>, MercuryVerifierKey<Bn254>) = mercury_srs.trim(3);

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let hkzg_srs = HyperKZGSRS::setup(&mut rng, 8);
        let (hkzg_pk, hkzg_vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = hkzg_srs.trim(8);

        // G1 SRS check
        for i in 0..(num_vars as f64 / 2.0).ceil() as usize{
            assert_eq!(hkzg_pk.kzg_pk.g1_powers()[i], mercury_pk.srs.0.kzg_srs.g1_powers[i], "failed at index {i}")
        }
        // G2 SRS check
        for i in 0..(num_vars as f64 / 2.0).ceil() as usize{
            assert_eq!(hkzg_pk.kzg_pk.g2_powers()[i], mercury_pk.srs.0.kzg_srs.g2_powers[i], "failed at index {i}")
        }

        // AFGHO SRS
        assert_eq!(mercury_pk.srs.0.kzg_srs.g2_powers[0], mercury_pk.srs.0.afgho_srs[0], "failed at index 0");

        let m = 1 << ((num_vars as f64 / 2.0).ceil() as usize);
        for i in 1..mercury_pk.srs.0.afgho_srs.len(){
            assert_eq!(mercury_pk.srs.0.kzg_srs.g2_powers[m * i - 1], mercury_pk.srs.0.afgho_srs[i], "failed at index {i}")
        }

    }
}