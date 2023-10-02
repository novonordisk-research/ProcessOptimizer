import csv
import math
import numpy as np

from .model_system import ModelSystem
from ..space import Real

#Utility-functions to calculate delta-e (best thought of as a measure of color difference). Code used is from https://github.com/kaineyb/deltae,
#please see the license file in the same repo for more information.
#TODO: Move this to a separate file and import it here or make it into a dependency.
def delta_e_1976(Lab1, Lab2):
    """
    Takes Lab values as a dictionary and outputs a DeltaE1976 calculation

    Example Dictionarys:
    Lab1 = {'L': 50.00, 'a': 2.6772, 'b': -79.7751}
    Lab2 = {'L': 50.00, 'a': 0.00, 'b': -82.7485}
    """

    delL = Lab1['L'] - Lab2['L']
    dela = Lab1['a'] - Lab2['a']
    delb = Lab1['b'] - Lab2['b']
    result = math.sqrt(delL * delL + dela * dela + delb * delb)
    return result

def delta_e_2000(Lab1, Lab2, verbose=False, test=False, formula='Rochester'):
    """
    Takes Lab values as a dictionary and outputs a DeltaE2000 calculation

    Example Dictionarys:
    Lab1 = {'L': 50.00, 'a': 2.6772, 'b': -79.7751}
    Lab2 = {'L': 50.00, 'a': 0.00, 'b': -82.7485}

    # verbose=True

    Prints out all of the calculations that comes up with the deltaE2000.

    # test=True

    Returns all of the calculations that comes up with the deltaE2000 in the below order:
    a1Prime, a2Prime, c1Prime, c2Prime, h1Prime, h2Prime, hBarPrime, g, t, sL, sC, sH, rT, DE2000
    Rounded to 4 decimal places, as that was what the test data set was rounded to. 

    formula kwarg can be 'Rochester' or 'Bruce'.

    Rochester uses a different calculation for hPrime, h1Prime, h2Prime and hBarPrime than Bruce
    Read the white paper by Gaurav Sharma, Wencheng Wu and Endul N. Dala (http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf)
    """

    kL = 1.0
    kC = 1.0
    kH = 1.0
    lBarPrime = 0.5 * (Lab1['L'] + Lab2['L'])
    c1 = math.sqrt(Lab1['a'] * Lab1['a'] + Lab1['b'] * Lab1['b'])
    c2 = math.sqrt(Lab2['a'] * Lab2['a'] + Lab2['b'] * Lab2['b'])
    cBar = 0.5 * (c1 + c2)
    cBar7 = cBar**7
    g = 0.5 * (1.0 - math.sqrt(cBar7 / (cBar7 + 25**7)))  # 25**7 or 6103515625

    a1Prime = Lab1['a'] * (1.0 + g)
    a2Prime = Lab2['a'] * (1.0 + g)

    c1Prime = math.sqrt(a1Prime * a1Prime + Lab1['b'] * Lab1['b'])
    c2Prime = math.sqrt(a2Prime * a2Prime + Lab2['b'] * Lab2['b'])

    cBarPrime = 0.5 * (c1Prime + c2Prime)

    if formula == 'Rochester':
        # Rochester hPrime
        def hPrime(B_i_star, aiPrime):
            if B_i_star == 0 and aiPrime == 0:
                return 0
            else:
                hPrime = math.atan2(B_i_star, aiPrime) * 180 / math.pi
                if hPrime < 0:
                    hPrime += 360
                return hPrime

        # Gets h1 and h2 Primes
        h1Prime = hPrime(Lab1['b'], a1Prime)
        h2Prime = hPrime(Lab2['b'], a2Prime)

        # START Rochester hBarprime
        if abs(h1Prime - h2Prime) <= 180 and c1Prime * c2Prime != 0:
            hBarPrime = (h1Prime + h2Prime) / 2

        elif abs(h1Prime - h2Prime) > 180 and (h1Prime + h2Prime) < 360 and c1Prime * c2Prime != 0:
            hBarPrime = (h1Prime + h2Prime + 360) / 2

        elif abs(h1Prime - h2Prime) > 180 and (h1Prime + h2Prime) >= 360 and c1Prime * c1Prime != 0:
            hBarPrime = (h1Prime + h2Prime - 360) / 2

        elif (c1Prime * c2Prime) == 0:
            hBarPrime = (h1Prime + h2Prime)
        # END Rochester hBarprime

    elif formula == 'Bruce':

        # Bruces hPrimes
        h1Prime = (math.atan2(Lab1['b'], a1Prime) * 180.0) / math.pi

        if (h1Prime < 0.0):
            h1Prime += 360.0

        h2Prime = (math.atan2(Lab2['b'], a2Prime) * 180.0) / math.pi

        if (h2Prime < 0.0):
            h2Prime += 360.0

        # Bruces hBarPrime
        hBarPrime = 0.5 * (h1Prime + h2Prime + 360.0) if abs(h1Prime -
                                                             h2Prime) > 180.0 else 0.5 * (h1Prime + h2Prime)

    t = 1.0 - 0.17 * math.cos((math.pi * (hBarPrime - 30.0)) / 180.0) + 0.24 * math.cos((math.pi * (2.0 * hBarPrime)) / 180.0) + \
        0.32 * math.cos((math.pi * (3.0 * hBarPrime + 6.0)) / 180.0) - \
        0.2 * math.cos((math.pi * (4.0 * hBarPrime - 63.0)) / 180.0)

    if (abs(h2Prime - h1Prime) <= 180.0):
        dhPrime = h2Prime - h1Prime

    else:
        dhPrime = h2Prime - h1Prime + \
            360.0 if h2Prime <= h1Prime else h2Prime - h1Prime - 360.0

    dLPrime = Lab2['L'] - Lab1['L']
    dCPrime = c2Prime - c1Prime
    dHPrime = 2.0 * math.sqrt(c1Prime * c2Prime) * \
        math.sin((math.pi * (0.5 * dhPrime)) / 180.0)

    sL = 1.0 + (0.015 * (lBarPrime - 50.0) * (lBarPrime - 50.0)) / \
        math.sqrt(20.0 + (lBarPrime - 50.0) * (lBarPrime - 50.0))

    sC = 1.0 + 0.045 * cBarPrime
    sH = 1.0 + 0.015 * cBarPrime * t

    dTheta = 30.0 * math.exp(-((hBarPrime - 275.0) / 25.0)
                             * ((hBarPrime - 275.0) / 25.0))

    cBarPrime7 = cBarPrime**7

    rC = math.sqrt(cBarPrime7 / (cBarPrime7 + 6103515625.0))
    rT = -2.0 * rC * math.sin((math.pi * (2.0 * dTheta)) / 180.0)

    DE2000 = math.sqrt((dLPrime / (kL * sL)) * (dLPrime / (kL * sL)) + (dCPrime / (kC * sC)) * (dCPrime / (kC * sC)) +
                       (dHPrime / (kH * sH)) * (dHPrime / (kH * sH)) + (dCPrime / (kC * sC)) * (dHPrime / (kH * sH)) * rT)

    # If arbitury Keyword arg verbose=True then print out the below
    if verbose == True:

        decoration = '-'*20

        print(decoration)
        print('LAB Value Input')
        print(decoration)
        print(f"LAB1: {Lab1['L']}, {Lab1['a']}, {Lab1['b']}")
        print(f"LAB2: {Lab2['L']}, {Lab2['a']}, {Lab2['b']}")
        print(decoration)
        print('Outputs')
        print(decoration)
        print(f'a1Prime: {round(a1Prime, 4)}')
        print(f'a2Prime: {round(a2Prime, 4)}')
        print(decoration)
        print(f'cBar: {round(cBar, 4)}')
        print(f'cBar7: {round(cBar7, 4)}')
        print(decoration)
        print(f'c1Prime: {round(c1Prime, 4)}')
        print(f'c2Prime: {round(c2Prime, 4)}')
        print(decoration)
        print(f'h1Prime: {round(h1Prime, 4)}')
        print(f'h2Prime: {round(h2Prime, 4)}')
        print(decoration)
        print(f'(abs)h1Prime - h2Prime: {round(abs(h1Prime - h2Prime), 4)}')
        print(decoration)
        print(f'hBarPrime: {round(hBarPrime, 4)}')
        print(f'g: {round(g, 4)}')
        print(f't: {round(t, 4)}')
        print(f'sL: {round(sL, 4)}')
        print(f'sC: {round(sC, 4)}')
        print(f'sH: {round(sH, 4)}')
        print(f'rT: {round(rT, 4)}')
        print(decoration)
        print(f'DE2000: {round(DE2000, 4)}')

    if test == True:
        a1Prime = round(a1Prime, 4)
        a2Prime = round(a2Prime, 4)
        c1Prime = round(c1Prime, 4)
        c2Prime = round(c2Prime, 4)
        h1Prime = round(h1Prime, 4)
        h2Prime = round(h2Prime, 4)
        hBarPrime = round(hBarPrime, 4)
        g = round(g, 4)
        t = round(t, 4)
        sL = round(sL, 4)
        sC = round(sC, 4)
        sH = round(sH, 4)
        rT = round(rT, 4)
        DE2000 = round(DE2000, 4)

        return a1Prime, a2Prime, c1Prime, c2Prime, h1Prime, h2Prime, hBarPrime, g, t, sL, sC, sH, rT, DE2000

    else:
        return DE2000