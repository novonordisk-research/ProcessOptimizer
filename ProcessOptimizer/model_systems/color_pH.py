import csv
import math
import numpy as np

from typing import List
from .model_system import ModelSystem
from ..space import Integer


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


def find_closest_match(file_name, target_vector, target_columns):
    # make docstring for this function
    '''
    file_name: name of the csv file
    target_vector: vector of target values
    target_columns: list of column names for the target values
    
    returns: row with the closest match to the target vector
    '''
    data = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append(row)

    # Convert data to numpy array for easier manipulation
    data = np.array(data)

    # Get the indices of the target columns
    target_indices = [headers.index(col) for col in target_columns]

    # Extract the data for the target columns
    target_data = data[:, target_indices].astype(float)

    # Calculate the absolute differences for column 'B'
    differences_B = np.abs(target_data[:, 0] - target_vector[0])

    # Get the minimum difference for column 'B'
    min_difference_B = np.min(differences_B)

    # Get all rows where the difference for column 'B' is equal to the minimum difference
    rows_with_min_difference_B = data[differences_B == min_difference_B]

    # Extract the data for column 'C' from these rows
    data_C = rows_with_min_difference_B[:, target_indices[1]].astype(float)

    # Calculate the absolute differences for column 'C'
    differences_C = np.abs(data_C - target_vector[1])

    # Get the index of the minimum difference for column 'C'
    min_difference_C_index = np.argmin(differences_C)

    # Return the row with the minimum difference for column 'C'
    return rows_with_min_difference_B[min_difference_C_index]


def color_finder_dict(file_name, target_well):
    '''
    file_name: name of the csv file
    target_well: well number of the target well
    
    returns: dictionary with the L,a,b values for the target well
    '''
    data = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append(row)

    # Convert data to numpy array for easier manipulation
    data = np.array(data)

    # Get the indices of the target columns
    target_indices = [headers.index(col) for col in ['Well','L','A','B']]

    # Extract the data for the target columns
    target_data = data[:, target_indices]
    
    # Take the row in which the 'Well' column matches the target well
    target_row = target_data[target_data[:,0] == target_well]
    
    #Transform the values in 'L','A','B' to floats
    target_row = target_row[0][1:].astype(float)
    
    #Return the values in 'L','A','B' as a dictionary
    return {'L': target_row[0], 'a': target_row[1], 'b': target_row[2]}


def color_difference(file_name, well1, well2):
    '''
    file_name: name of the csv file
    well1: well number of the first well
    well2: well number of the second well
    
    returns: delta-e value between the two wells
    '''
    #Find the lab color using color_finder and calculate the difference between the two colors. use the method delta_E from the colour package.
    color1 = color_finder_dict(file_name, well1)
    color2 = color_finder_dict(file_name, well2)
    #return colour.delta_E(color1, color2, method='CIE 2000')
    return delta_e_2000(color1, color2, verbose=False, test=False, formula='Bruce')


def score(coordinates: List[int], evaluation_target='F8'):
    """
    coordinates: list of coordinates for a single experiment
    evaluation_target: well number of the target well. Default is F8 (because green is good for the eyes)
    
    returns: delta-e value between the two wells
    
    Takes a list of coordinates for a single experiment and compare to the color in target cell. It then returns the delta-e value between the two wells.
    With the default value of F8 as target, the right recipy will be 50% acid and 30uL Indicator [50,30].
    """
    file_name = './data/color_pH_data.csv'
    data_lookup_position = find_closest_match(file_name, coordinates, ['percent_acid', 'Indicator'])[0]
    evaluation = color_difference(file_name, data_lookup_position, evaluation_target)
    return evaluation
    
#Create model system
color_pH_no_noise = ModelSystem(
    score,
    space = [Integer(30, 85, name='percent_acid'),
             Integer(5, 40, name='Indicator'),
            ],
    noise_model = None,
    true_min = 0,
)

color_pH = ModelSystem(
    score,
    space = [Integer(30, 85, name='percent_acid'),
             Integer(5, 40, name='Indicator'),
            ],
    noise_model = {"model_type": "proportional", "noise_size": 0.1},
    true_min = 0,
)