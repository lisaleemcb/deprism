import survey

from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as const

nu_21cm = 1420 * u.MHz
nu_CII = 1.900539e12 * u.Hz # Hz
nu_CO = 115.271203e9 * u.Hz # Hz
nu_OIII = 1e9 * u.Hz # Hz

lambda_21cm = nu_21cm.to(u.cm, equivalencies=u.spectral())
lambda_CII = nu_CII.to(u.um, equivalencies=u.spectral()) #158 um micrometers
lambda_CO = nu_CO.to(u.mm, equivalencies=u.spectral())
lambda_OIII = nu_OIII.to(u.cm, equivalencies=u.spectral())

specs_CCATp = {'sigma_pix': 0.86 * (u.MJy * u.s**(1/2) / u.steradian),
               'N_det': 20 * u.dimensionless_unscaled,
               'theta_FWMH': 46.0 * u.arcsec,
               'nu_obs_min': 200.0 * u.GHz,
               'nu_obs_max': 300.0 * u.GHz,
               'delta_nu': 2.5 * u.GHz,
               't_obs': 3400 * u.hr,
               'Omega_surv': 1.7 * u.degree**2,
               'AGN Source': 'COSMOS'}

specs_HERA = {'sigma_pix': None,
               'N_det': None,
               'theta_FWMH': None,
               'min_baseline': 14.6 * u.m,
               'max_baseline': 140 * u.m,
               'D_dish': 140 * u.m,
               'nu_obs_min': 100.0 * u.GHz,
               'nu_obs_max': 200.0 * u.GHz,
               'delta_nu': 97.8 * u.kHz,
               'B_nu': 100 * u.GHz,
               't_obs': None,
               'S_A': 1440 * u.degree**2,
               'AGN Source': None,
               'N_pol': 2,
               'T_sys': 400,
               't_int': 1000}

# D_dish is specified due to coding laziness. Should be max_baseline
specs_HERA_future = {'sigma_pix': None,
               'N_det': None,
               'theta_FWMH': None,
               'min_baseline': 14.6 * u.m,
               'max_baseline': 5000 * u.m,
               'D_dish': 5000 * u.m,
               'nu_obs_min': 100.0 * u.GHz,
               'nu_obs_max': 200.0 * u.GHz,
               'delta_nu': 97.8 * u.kHz,
               'B_nu': 100 * u.GHz,
               't_obs': None,
               'S_A': 1440 * u.degree**2,
               'AGN Source': None,
               'N_pol': 2,
               'T_sys': 400,
               't_int': 1000}

# FYST-like Stage II (formerly CCAT-prime) from Padmanabhan et al. (2021)
specs_StageII = {'D_dish': 9 * u.m,
                'delta_nu': 400 * u.MHz,
                'N_spec_eff': 1,
                'S_A': 100 * u.degree**2,
                'sigma_N': 4.84e4 * u.Jy * u.s**(.5) / u.steradian,
                'B_nu': 40 * u.GHz,
                't_obs': 2000 * u.hr,
                'Survey_Bandwidth_start': 212 * u.GHz,
                'Survey_Bandwidth_finish': 428 * u.GHz}

# FYST-like State III/IV from Padmanabhan et al. (2021)
specs_StageIII = {'D_dish': 9 * u.m,
                'delta_nu': 400 * u.MHz,
                'N_spec_eff': 16000,
                'S_A': 100 * u.degree**2,
                'sigma_N': 2.1e5 * u.Jy * u.s**(.5) / u.steradian,
                'B_nu': 40 * u.GHz,
                't_obs': 2000 * u.hr,
                'Survey_Bandwidth_start': 212 * u.GHz,
                'Survey_Bandwidth_finish': 428 * u.GHz}

# EXCLAIM-like from Padmanabhan et al. (2021)
specs_EXCLAIM = {'D_dish': .74 * u.m,
                'delta_nu': 1000 * u.MHz,
                'N_spec_eff': 30,
                'S_A': 100 * u.degree**2,
                'sigma_N': 3.0e5 * u.Jy * u.s**(.5) / u.steradian,
                'B_nu': 40 * u.GHz,
                't_obs': 72 * u.hr,
                'Survey_Bandwidth_start': 420 * u.GHz,
                'Survey_Bandwidth_finish': 540 * u.GHz}

specs_future = {'D_dish': 3 * u.m,
                'delta_nu': 300 * u.MHz,
                'N_spec_eff': 50,
                'S_A': 16 * u.degree**2,
                'sigma_N': 1.6e5 * u.Jy * u.s**(.5) / u.steradian,
                'B_nu': 100 * u.GHz,
                't_obs': 4000 * u.hr,
                'Survey_Bandwidth_start': 250 * u.GHz,
                'Survey_Bandwidth_finish': 900 * u.GHz}
