import pandas as pd
import numpy as np
import re
import pyodbc
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\christopher.johnson\\Documents\\')
import ctj_tools as ros
import datetime
from dateutil.relativedelta import relativedelta

# ML packages
from sklearn.preprocessing import StandardScaler


import requests
import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\christopher.johnson\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

from difflib import get_close_matches
import textwrap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



class InspectionClassifications():
    
    """
    Preprocessing class for predicting inspection severity
    """
    
    def __init__ (self,date_):

        fda_inspections_path = "Files\\Downloads\\FDAInspectionDetails."+date_+".xlsx"
        fda_citations_path = "Files\\Downloads\\InspectionCitations."+date_+".xlsx"
        fda_warningletters_path = "Files\\Downloads\\WarningLetters."+date_+".xlsx"
        fda_form483_path = "Files\\Downloads\\FDAForm483."+date_+".xlsx"
        fda_recalls_path = "Files\\Downloads\\Recalls."+date_+".xlsx"

        
       
        def get_classification(x):
            m = re.search(r'\((.*?)\)',x)
            if m:
                return m.group(1)
            else:
                return None

        # Inspections
        fda_insp_details_df = pd.read_excel(fda_inspections_path)
        # make column names not annoying
        fda_insp_details_df = ros.fix_columns(fda_insp_details_df)
        fda_insp_details_df.columns = ['FEINumber', 'LegalName', 'City', 'State', 'Zip', 'CountryArea',
            'FiscalYear', 'InspectionID', 'PostedCitations', 'InspectionEndDate',
            'Classification', 'ProjectArea', 'ProductType','AdditionalDetails',
            'FMD145Date']
        cols_keep = ['FEINumber', 'LegalName', 'City', 'State', 'Zip', 'CountryArea',
            'FiscalYear', 'InspectionID', 'PostedCitations', 'InspectionEndDate',
            'Classification', 'ProjectArea', 'ProductType']
        fda_insp_details_df = fda_insp_details_df[cols_keep]
        fda_insp_details_df['Classification'] = fda_insp_details_df['Classification'].apply(lambda x: get_classification(x))

        # Citations
        fda_cit_df = ros.fix_columns(pd.read_excel(fda_citations_path))
        fda_cit_df['Act/CFRNumber'] = fda_cit_df['Act/CFRNumber'].fillna(' ')
        fda_cit_df.columns = ['InspectionID', 'FEINumber', 'LegalName', 'InspectionEndDate',
            'ProgramArea', 'ActCFRNumber', 'ShortDescription', 'LongDescription']

        # Warning Letters
        warning_letters_df = pd.read_excel(fda_warningletters_path)
        warning_letters_df.columns = ['FEINumber', 'LegalName', 'State', 'CountryArea', 'ProductType',
            'ActionTakenDate', 'ActionType', 'CaseInjunctionID']
        
        # 483s
        form_df = ros.fix_columns(pd.read_excel(fda_form483_path))

        # Recalls
        recalls_df = pd.read_excel(fda_recalls_path)
        recalls_df.columns = ['FEINumber','FirmName','ProductType','ProductClassification','Status','DistributionPattern','FirmCity','FirmState','FirmCountry','CenterClassificationDate','Reason','Product','EventID','EventClassification','ProductID','Center','RecallDetails']
        recalls_df = recalls_df[recalls_df.ProductType == 'Drugs']
        recalls_df = recalls_df[recalls_df.FEINumber !='-']
        recalls_df['FEINumber'] = recalls_df['FEINumber'].astype('int64')

        self.fda_insp_details_df = fda_insp_details_df
        self.fda_cit_df = fda_cit_df
        self.warning_letters_df = warning_letters_df
        self.form_df = form_df
        self.recalls_df = recalls_df
        self.fei_ros_df = fei_ros_df

    def preprocess(self,citation_start_date, citation_end_date):
        
        fda_cit_df = self.fda_cit_df
        fda_insp_details_df = self.fda_insp_details_df
        fda_cit_drugs_df = fda_cit_df[fda_cit_df.ProgramArea=='Drugs']
        data_df = fda_cit_drugs_df.merge(fda_insp_details_df[['InspectionID','Classification']].drop_duplicates(),
                                        how='left',
                                        on='InspectionID')
        data_df['Year'] = data_df['InspectionEndDate'].apply(lambda x: x.year)



        data_trim_df = data_df.copy()
        data_trim_df = data_trim_df.drop('Classification',axis=1).drop_duplicates().merge(data_trim_df[['InspectionID','Classification']].drop_duplicates().groupby(['InspectionID'])['Classification'].sum().reset_index(),
                        how = 'left',
                        on = 'InspectionID')
        data_trim_df = data_trim_df.drop_duplicates()
        data_trim_df.columns = ['InspectionID', 'FEINumber', 'LegalName', 'InspectionEndDate',
        'ProgramArea', 'ActCFRNumber', 'ShortDescription', 'LongDescription','Year',
        'Classification']

        def fix_classification(x):
            if x.count('OAI') or x.count('Recall') > 0:
                return 'OAI'
            elif x.count('VAI') > 0:
                return 'VAI'
            else:
                return x
                
        #data_trim_df['Classification'] = data_trim_df['Classification'].apply(lambda x: fix_classification(x))
        data_trim_df = data_trim_df[data_trim_df['Classification'].isin(['OAI','VAI'])]
        data_trim_df = data_trim_df.merge(fda_insp_details_df[['FEINumber','CountryArea']].drop_duplicates(),
                                    how = 'left',
                                    on = 'FEINumber')

        act_df = data_trim_df[(data_trim_df.InspectionEndDate < citation_end_date) & (data_trim_df.InspectionEndDate >= citation_start_date)][['InspectionID','ActCFRNumber','Classification']]
        act_df['Values'] = len(act_df)*[1]

        act_df = act_df.pivot_table(index='Classification',columns='ActCFRNumber',values='Values',aggfunc='sum').fillna(0).T
        act_df['Total'] = act_df.T.sum().values
        act_df['OAI_weighted'] = act_df['OAI'] / act_df['Total']
        #act_df['OAI_weighted'] = [1]*len(act_df)

        data_trim_df = data_trim_df.merge(act_df,how='left',on='ActCFRNumber')
        #data_trim_df['OAI_weighted'].isnull().sum()

        input_df = data_trim_df.pivot_table(index='InspectionID',columns='ActCFRNumber',values='OAI_weighted').fillna(0)
        input_df['CitationScore'] = input_df.T.sum().values
        ss = StandardScaler()
        ss.fit(input_df[['CitationScore']])

        input_df = input_df.merge(data_df.groupby('InspectionID')['ActCFRNumber'].nunique(),how='left',on='InspectionID')
        ss_act = StandardScaler()
        ss_act.fit(input_df[['ActCFRNumber']])

        input_df['CitationScore_mean'] = input_df['CitationScore']/input_df['ActCFRNumber']

        ss_ccw = StandardScaler()
        ss_ccw.fit(input_df[['CitationScore_mean']])
        
        
        input_df['ActCFRNumbernoNorm'] = input_df['ActCFRNumber'] 
        input_df['CitationScorenoNorm'] =input_df['CitationScore']
        input_df['CitationScore_meannoNorm'] = input_df['CitationScore_mean']

        input_df['ActCFRNumber'] = ss_act.transform(input_df[['ActCFRNumber']])
        input_df['CitationScore'] = ss.transform(input_df[['CitationScore']])
        input_df['CitationScore_mean'] = ss_ccw.transform(input_df[['CitationScore_mean']])

        def encode_country(x):
            if x == "United States":
                return 1
            elif x == 'India':
                return -1
            else:
                return 0

        input_df = input_df.merge(data_trim_df[['InspectionID','CountryArea']].drop_duplicates(),how='left',on='InspectionID')
        input_df['CountryArea'] = input_df['CountryArea'].apply(lambda x: encode_country(x))

        input_df = input_df.merge(data_trim_df[['InspectionID','Year']].drop_duplicates(),
                            how = 'left',
                            on = 'InspectionID')
        input_df['Year'] = input_df['Year'].apply(lambda x: 1 if x > 2020 else 0)

        input_df = input_df.merge(data_trim_df[['InspectionID','Classification']].drop_duplicates(),
                            how = 'left',
                            on = 'InspectionID')
        input_df['Classification'] = input_df['Classification'].apply(lambda x: 1 if x == 'OAI' else 0)
        input_df = input_df.set_index('InspectionID')

        #top_200_features = list(set(abs(input_df.drop(['ActCFRNumbernoNorm','CitationScorenoNorm','CitationScore_meannoNorm'],axis=1).corr()['Classification']).sort_values(ascending=False).head(201).index) - set(['Classification']))
        #input_trim_df = input_df[['Classification']+top_200_features]
        # reorder columns
        cols = list(input_df.drop(['ActCFRNumbernoNorm','CitationScorenoNorm','CitationScore_meannoNorm'],axis=1).columns)
        cols.sort()
        input_trim_df = input_df[cols]

        return input_trim_df, input_df, data_df, act_df, [ss_act,ss,ss_ccw]

def has_numbers(inputString):

    """
    Needed in 'pdf_to_text_crop'
    
    Returns True if string contains numbers and index of number
    """
    
    out_bool =  any(char.isdigit() for char in inputString)

    out_ind = ''
    if out_bool == True:

        for i in range(len(inputString)):

            char = inputString[i]
            if char.isdigit():
                
                out_ind = i
    return out_bool, out_ind


def pdf_to_text_crop(filePath):

    """
    Converts 483 PDF into 
        1. Full string
        2. List of images of each page
        3. List of observations reported
    """
    
    doc = convert_from_path(filePath)
    path, fileName = os.path.split(filePath)
    fileBaseName, fileExtension = os.path.splitext(fileName)

    total_txt = ''
    total_page_data = []
    for page_number, page_data in enumerate(doc):
        if page_number == 0:
            data = pytesseract.image_to_data(page_data,output_type='dict')
            data_df = pd.DataFrame(data = zip(data['text'],data['left'],data['top'],data['width'],data['height']), columns = ['text','left','top','width','height'])
            try:
                crop_top = data_df[data_df.text == 'document']['top'].values[0]
            except:
                crop_top = data_df[data_df.text == 'DOCUMENT']['top'].values[0]
            
            crop_bot = data_df[data_df.text == 'EMPLOYEE(S)']['top'].values[0]+2
        page_data = page_data.crop((0,crop_top,2000,crop_bot))
            
        txt = pytesseract.image_to_string(page_data).encode("utf-8")
        txt = str(txt)
        
        txt = txt.replace('\\n','')
        total_txt+=str(txt)+'\n'
        total_page_data.append(page_data)
    
    # collect observations
    total_txt = total_txt.upper()
    obs = total_txt.split('OBSERVATION')
    obs_list = []
    for ob in obs:
        observation = ob.split('SPECIFICALLY')[0]
        out_bool, out_ind = has_numbers(observation[:4])
        if out_bool:
            
          # observation = observation.split(":")[1]
            obs_list.append(observation[int(out_ind+1):])
    
    
    return total_txt, total_page_data, obs_list

def get_citations_from_observations(obs_list,data_df):

    """ 
    Args: list of observations from 483
        
    Returns: CFR citation numbers
    """
    cfr_list, new_obs_list = [],[]

    data_df['LongDescription'] = data_df['LongDescription'].apply(lambda x: x.upper())

    for obs in obs_list:

        matches = get_close_matches(obs,data_df["LongDescription"].unique())
        cfr_matches = data_df[data_df['LongDescription'].isin(matches)]['ActCFRNumber'].unique()
        cfr = cfr_matches[0]
        print("Observation:",obs)
        print("Matches:",cfr_matches,matches)

        if len(cfr_matches)>1:
            
            trim_df = data_df[data_df['LongDescription'].isin(matches)][['ActCFRNumber','LongDescription']].drop_duplicates().reset_index().drop('index',axis=1)
            print('********************************************************')
            print("Multiple CFR numbers corresponding to observation:",obs)
            print(trim_df.to_string())
            print('********************************************************')
            print('Which entry?')
            index_ = int(input())

            cfr = trim_df['ActCFRNumber'].iloc[index_]
            new_obs =  trim_df['LongDescription'].iloc[index_]

        else:
            
            new_obs = matches[0]
        
        cfr_list.append(cfr)
        new_obs_list.append(new_obs)

    return cfr_list, new_obs_list

def create_table(act_df,cit_list, obs_list):

    """
    Creates report from 483
    """

    output_table_df = act_df.loc[cit_list].drop('VAI',axis=1)
    output_table_df[['OAI','Total']] = output_table_df[['OAI','Total']].astype('int32')
    #output_table_df['OAI_weighted'] = output_table_df['OAI_weighted'].apply(lambda x: f'{x*100:.1f}%')
    output_table_df.columns = ['OAI','Total','OAI(%)']
    output_table_df['Observations'] = obs_list
    output_table_df = output_table_df[['Observations','OAI','Total','OAI(%)']]
    output_table_df['Observations'] = output_table_df['Observations'].apply(lambda x: textwrap.fill(x,width=60))
    output_table_df = output_table_df.reset_index()
    
    def make_pretty(styler):
        
        styler.set_caption("FDA History")
        styler.background_gradient(axis=1, vmin=0., vmax=0.65, cmap="RdYlGn_r",subset = ['OAI(%)'])
        return styler

    return output_table_df.style.pipe(make_pretty)

def create_cit_plot(act_df,cit_list, obs_list,input_full_df):

    """
    Creates citation plot from 483
    Args: citation score df, citation list, observation list, full input

    """

    # create plot
    fig, ax = plt.subplots(figsize=(10,5))
    

    cit_num = len(cit_list)
    cit_score = act_df.loc[cit_list]['OAI_weighted'].sum()
    ax.set_title('Citations = '+str(len(cit_list)),fontsize=20)
    input_full_df[(input_full_df.ActCFRNumbernoNorm == cit_num) & (input_full_df.Classification == 1) ]['CitationScorenoNorm'].hist(ax=ax,bins=30,color='red',stacked=True, density=True,alpha = 0.75,label = 'OAI')
    input_full_df[(input_full_df.ActCFRNumbernoNorm == cit_num) & (input_full_df.Classification == 0)]['CitationScorenoNorm'].hist(ax=ax,bins=30,color='purple',density=True,alpha=0.75,label='no OAI',stacked=True)
    ax.axvline(x=cit_score,linestyle='dashed')
    ax.set_xlabel('Citation Score',fontsize=15)
    
    ax.legend(loc = 'upper left')

def create_new_input(year,countryarea, ss_list, input_df,cit_list,act_df):

    """
    Create's new input for new inspection

    Args: Year = (0 or 1), CountryArea = (-1,0,1), standard scaler list, input df, citation list, citation score df
    Returns: single row input for models
    
    """

    #year, countryarea = 1,-1
    ss_act, ss, ss_ccw = ss_list
    cols = list(input_df.drop('Classification',axis=1).columns)
    new_input = pd.DataFrame(data = np.zeros(shape=(1,len(cols))),columns = cols)
    new_input[cit_list] = act_df.loc[cit_list]['OAI_weighted']
    new_input['CitationScore'] = [act_df.loc[cit_list]['OAI_weighted'].sum()]
    new_input['ActCFRNumber'] = [len(cit_list)]
    new_input['CitationScore_mean'] = [act_df.loc[cit_list]['OAI_weighted'].sum()/len(cit_list)]
    new_input['ActCFRNumber'] = ss_act.transform(new_input[['ActCFRNumber']])
    new_input['CitationScore'] = ss.transform(new_input[['CitationScore']])
    new_input['CitationScore_mean'] = ss_ccw.transform(new_input[['CitationScore_mean']])
    new_input['CountryArea'] = countryarea
    new_input['Year'] = year

    return new_input