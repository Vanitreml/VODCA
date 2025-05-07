import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import ast


def data_evaluation(path, **kwargs):
    """
    evaluates the data

    Parameters
    ----------
    path : str
        path to the evaluated folder
    **kwargs : TYPE
        optional arguments for the calculation of Nm

    Returns
    -------
    gesamtdaten_Nm : dataframe
    'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets', 'frozen_fraction, 'Nm'

    """

    csv_dateien = glob.glob(os.path.join(path, "*.csv"))
    dataframes = []


    for datei in csv_dateien:

        if 'droplets_' in datei:
            df = pd.read_csv(datei, header='infer', sep=',')
            dataframes.append(df)
            df = pd.concat(dataframes)

    df = df.sort_values(by="temperature")
    df.set_index('temperature', inplace=False)
    df_af = calculate_already_frozen(df)
    df_ff = calculate_frozen_fraction(df_af)

    gesamtdaten_Nm = calculate_Nm(df_ff, **kwargs)
    save_path = os.path.join(
        path, f'overall_evaluation_{os.path.basename(path)}.csv')
    gesamtdaten_Nm.to_csv(save_path, index=False)

    plot_Nm(gesamtdaten_Nm, path)

    return gesamtdaten_Nm


def calculate_already_frozen(gesamtdaten):
    """
    sums up the number of frozen droplets

    Parameters
    ----------
    gesamtdaten : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets'

    Returns
    -------
    gesamtdaten : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets'

    """
    already_frozen_df = pd.DataFrame()
    already_frozen_list = []
    for i in range(len(gesamtdaten)):
        dasisteineSerie = gesamtdaten.iloc[0:i+1, 1].sum()
        already_frozen_list.append(dasisteineSerie)

    already_frozen_df['Already_frozen'] = already_frozen_list
    already_frozen_df.index = gesamtdaten.index
    gesamtdaten = gesamtdaten.assign(
        Already_frozen=already_frozen_df['Already_frozen'])

    return gesamtdaten


def runden_sig_stellen(liste, sig=2):
    """
    rundet eine Zahl auf eine gebenene Anzahl an signifikanten Stellen

    Parameters
    ----------
    liste : list
        containing digits to be rounded
    sig : int, optional
        the number of significant figures. The default is 2.

    Returns
    -------
    liste: list
        rounded digits

    """

    liste = np.array(liste, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):  
        factor = np.power(10, sig - np.floor(np.log10(np.abs(liste))) - 1)
        return np.round(liste * factor) / factor


def calculate_frozen_fraction(gesamtdaten_af):
    """
    calculates the frozen fraction

    Parameters
    ----------
    gesamtdaten_af : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets'

    Returns
    -------
    gesamtdaten_ff : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets', 'frozen_fraction'

    """

    frozen_fraction_df = pd.DataFrame()
    frozen_fraction_list = []
    last_index = len(gesamtdaten_af)
    total_number_of_droplets = gesamtdaten_af.iloc[last_index-1, 3]

    for i in range(len(gesamtdaten_af)):

        frozen_fraction_list.append(
            gesamtdaten_af.iloc[i, 3]/total_number_of_droplets)

    frozen_fraction_list = runden_sig_stellen(frozen_fraction_list, 4)
    frozen_fraction_df['Frozen_Fraction'] = frozen_fraction_list
    frozen_fraction_df.index = gesamtdaten_af.index

    gesamtdaten_ff = gesamtdaten_af.assign(
        frozen_fraction=frozen_fraction_df['Frozen_Fraction'])

    return gesamtdaten_ff


def calculate_Nm(gesamtdaten, d=1, a=1, b=1, V_method='mean'):
    """
    calculates Nm

    Parameters
    ----------
    gesamtdaten : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets', 'frozen_fraction'
    d : float, optional
        DESCRIPTION. The default is 1.
    a : flaot, optional
        DESCRIPTION. The default is 1.
    b : float, optional
        DESCRIPTION. The default is 1.
    V_method : TYPE, optional
        Calculation can be carried out considering the individual Volume of each droplet or the mean Volume of all droplets. The default is 'mean'.

    Returns
    -------
    gesamtdaten_Nm : TYPE
        DESCRIPTION.

    """

    Nm_df = pd.DataFrame()
    Nm_list = []

    for i in range(len(gesamtdaten)):
        if gesamtdaten.iloc[i, 4] != 1:
            if V_method == 'individually':
                V = calculate_volume(gesamtdaten.iloc[i, 2])
            if V_method == 'mean':
                V = calculate_volume_with_mean(gesamtdaten)
            Nm_list.append(-np.log(1-gesamtdaten.iloc[i, 4])*(d*a)/(V*b))
        else:
            Nm_list.append(0)
    Nm_list = runden_sig_stellen(Nm_list, 4)
    Nm_df['Nm'] = Nm_list
    Nm_df.index = gesamtdaten.index

    gesamtdaten_Nm = gesamtdaten.assign(Nm=Nm_df['Nm'])
    return gesamtdaten_Nm


def calculate_volume(gesamtdaten_ausschnitt):
    """
    calculates the volume of the individual droplets

    Parameters
    ----------
    gesamtdaten_ausschnitt : int
        radius of the droplet

    Returns
    -------
    V : float
         volume of the droplet

    """

    radii_list = gesamtdaten_ausschnitt[2:-2]
    radii_list = radii_list.split(' ')
    radii_list_int = []

    print(radii_list, gesamtdaten_ausschnitt)

    for i in range(len(radii_list)):
        radii_list_int.append(int(radii_list[i]))

    r = np.mean(radii_list_int)
    V = ((r/1000000)**3) * 4*np.pi/3
    return V


def sum_list_elements(string):
    """
    sums up all elements in a string

    Parameters
    ----------
    string : str


    Returns
    -------
            float
            sum of elements of the list

    """
    try:
        # replaces whitespaces with ','
        string = string.replace(" ", ",")
        # change datatype to list
        lst = ast.literal_eval(string)
        if isinstance(lst, list):
            return sum(lst)  # Summiere alle Elemente in der Liste
        return 0
    except Exception as e:
        print(e)
        return 0


def calculate_volume_with_mean(df):
    """
    calculates the mean volume

    Parameters
    ----------
    df : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets', 'frozen_fraction'

    Returns
    -------
    V : float
        mean Volume

    """

    # The elements were saved as np.arrays in the csv, they have to be unpacked 
    rs = df['radius frozen droplets'].apply(sum_list_elements)
    #calculates the mean Volume
    r_mean = rs.mean()
    #r_mean = 23.6
    V = ((r_mean/1000000)**3) * 4*np.pi/3
    return V


def plot_Nm(gesamtdaten_Nm, path):
    """
    plots the Nm data

    Parameters
    ----------
    gesamtdaten_Nm : dataframe
        'temperature', 'number of frozen droplets', 'radius frozen droplets', 'sum of already frozen droplets', 'frozen_fraction, 'Nm'
    path : str
        str containing the path to the evaluated folder.

    Returns
    -------
    None

    """
    plt.plot(-gesamtdaten_Nm.iloc[:, 0], gesamtdaten_Nm.iloc[:, 5], 'o')
    plt.yscale('log')
    plt.ylabel('Nm')
    plt.xlabel('temperature in °C')
    save_path = os.path.join(path, f'Nm_{os.path.basename(path)}.png')
    plt.savefig(save_path)
    plt.close()



