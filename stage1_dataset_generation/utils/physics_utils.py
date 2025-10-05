import numpy as np
def angular_separation(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.deg2rad, [ra1, dec1, ra2, dec2])
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2)**2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(a,0,1))))
