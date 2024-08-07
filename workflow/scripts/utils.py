import numpy as np
import scipy.constants as sc

vacuum_permittivity = sc.epsilon_0
gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

def ionic_strength(z, c):
    """Compute a system's ionic strength from charges and concentrations.

    Returns
    -------
    ionic_strength : float
        ionic strength ( 1/2 * sum(z_i^2*c_i) )
        [concentration unit, i.e. mol m^-3]
    """
    return 0.5*np.sum(np.square(z) * c)


def lambda_D(ionic_strength, temperature, relative_permittivity,
             vacuum_permittivity=vacuum_permittivity,
             gas_constant=gas_constant,
             faraday_constant=faraday_constant):
    """Compute the system's Debye length.

    Returns
    -------
    lambda_D : float
        Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [length unit, i.e. m]
    """
    return np.sqrt(
        relative_permittivity * vacuum_permittivity * gas_constant * temperature / (
                2.0 * faraday_constant ** 2 * ionic_strength))