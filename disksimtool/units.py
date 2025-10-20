from astropy import constants as c

class U:
    au = c.au.cgs.value
    pc = c.pc.cgs.value
    Msun = c.M_sun.cgs.value
    Lsun = c.L_sun.cgs.value
    G = c.G.cgs.value

units = U()