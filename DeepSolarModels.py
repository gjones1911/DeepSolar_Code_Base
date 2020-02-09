"""          This is a collection of different variables sets and a few data structures to make them  """
"""          easier to load into a model, """
from statistics import mean

import pandas as pd
#from _products.ML_Tools import load_tree_trunc_features
def load_tree_trunc_features(df=None, dffile=None, limit=.00, verbose=False):
    if df is None:
        df = pd.read_excel(dffile, usecols=['Variable', 'Imp_trunc'])

    df = df.loc[df['Imp_trunc'] >= limit, 'Variable']
    if verbose:
        print(list(df))
    return list(df)

def voted_list(vote_dic, thresh=2):
    rl = list()
    for f in vote_dic:
        if vote_dic[f] > 2:
            rl.append(f)
    return rl

def tally_var_votes(votes,):
    #from _products.utility_fnc import sort_dict
    import operator
    rd = {}
    for va in votes:
        for v in va:
            if v not in rd:
                rd[v] = 0
            rd[v] += 1

    return dict(sorted(rd.items(), key=operator.itemgetter(1), reverse=True))

popdenonly = ['Adoption', 'population_density', 'heating_fuel_coal_coke_rate',
              'education_bachelor', 'pop25_some_college_plus', 'travel_time_49_89_rate', 'education_master',
              'pop_female', '',
              ]

# ##############################################################
# ##############################################################
# ############TODO: Random Forest FS sets/simple sets  #####################
# ##############################################################
# ##############################################################

tallff = 0
#  0.7333350960892336, Good is now:
simple_set4 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'pop_over_65', 'hu_2000toafter_scld']

# Score: 0.735002341556155, Good is now:
simple_set10 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'incent_cnt_res_own',
                'travel_time_49_89_rate', 'education_doctoral']

# Score: 0.7352799456986788, Good is now:
simple_set9 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'heating_fuel_none', 'hu_med_val_scld',
               'age_55_64_rate']

# Score: 0.7355576655819634, Good is now:
simple_set11 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'hu_2000toafter_scld', 'pop_over_65',
                'heating_fuel_none_rate', 'education_less_than_high_school']

#  0.7358360798133626
simple_set3 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'hu_monthly_owner_costs_lessthan_1000dlrs', 'avg_electricity_retail_rate_scld', 'land_area_scld',
               'hu_vintage_1939toearlier_scld']

# Score: 0.7361141471600564, Good is now:
simple_set6 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'age_65_74_rate',
               'total_area', 'heating_fuel_gas_scld', 'hu_2000toafter_scld',
               'hu_monthly_owner_costs_greaterthan_1000dlrs', 'incentive_count_nonresidential']

# 0.7363917513990309
simple_set = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education',
              'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'avg_electricity_retail_rate_scld', 'heating_fuel_other']

# Score: 0.7376407100697606
simple_set7 =['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'pop_over_65',
              'hu_2000toafter_scld', 'incent_cnt_res_own', 'travel_time_10_19_rate', 'employ_rate_scld',
              'education_bachelor_rate', 'pop_med_age_scld', 'hh_med_income']

#  0.7377803512494823
simple_set2 = ['population_density_scld', 'E_DAYPOP', 'number_of_years_of_education_scld', 'total_area_scld',
               'age_65_74_rate', 'poverty_family_below_poverty_level_rate', 'heating_fuel_solar_scld',
               'travel_time_10_19_rate', 'occupation_finance_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
               'heating_fuel_coal_coke_scld', 'heating_fuel_none']

# e: 0.737853762194533, 0.21913339615387883
simple_set8 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'avg_electricity_retail_rate_scld', 'travel_time_10_19_rate']

# Score: 0.737853762194533, Good is now:
simple_set12 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
                'avg_electricity_retail_rate_scld', 'travel_time_10_19_rate']

# Score: 0.7384736961325862, Good is now: TODO: take out incentive residential
simple_set5 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education', 'incentive_count_residential',
               'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'avg_electricity_retail_rate_scld',
               'incentive_count_nonresidential', 'fam_med_income_scld', 'travel_time_10_19_rate', 'age_75_84_rate']

# Score: 0.739968571305989, Good is now:
simple_set13 = ['population_density_scld', 'E_DAYPOP', 'number_of_years_of_education_scld', 'heating_fuel_none',
                'total_area_scld', 'poverty_family_below_poverty_level_rate', 'occupation_finance_rate',
                'travel_time_10_19_rate', 'heating_fuel_solar_scld']

simple_set_avg = mean([0.7363917513990309, 0.7377803512494823, 0.7358360798133626, 0.7333350960892336,
                       0.7384736961325862,0.7361141471600564, 0.7376407100697606, 0.737853762194533,
                       0.7352799456986788, 0.7355576655819634, 0.737853762194533])

# ##############################################################
# ##############################################################
# ############TODO: Logistic Regression FS sets  ###############
# ##############################################################
# ##############################################################
R2set = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'heating_fuel_coal_coke_scld',
         'population_density_scld', 'hu_med_val_scld', 'pop_total_scld', 'hu_vintage_1939toearlier_scld',
         'education_bachelor_rate', 'incent_cnt_res_own', 'pop_under_18_scld', 'avg_inc_ebill_dlrs_scld',
         'housing_unit_median_gross_rent_scld', 'pop_over_65_scld', 'education_master_scld', 'age_median_scld',
         'heating_fuel_gas_scld', 'net_metering_hu_own', 'average_household_income_scld', 'avg_monthly_bill_dlrs_scld',
         'E_AGE17_scld', 'travel_time_49_89_rate','hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
         'education_bachelor_scld', 'education_master_rate', 'age_25_44_rate_scld',
         'education_doctoral_scld']

good_acc_lr = ['education_bachelor_scld', 'total_area_scld', 'pop_total_scld', 'population_density_scld',
               'hu_vintage_1939toearlier_scld', 'land_area_scld', 'net_metering_hu_own', 'housing_unit_count_scld',
               'own_popden_scld']
# .7178637200736648
good_acc_lr2 = ['education_bachelor_scld', 'travel_time_average_scld', 'heating_fuel_housing_unit_count_scld',
                'hu_med_val_scld', 'heating_fuel_coal_coke_scld', 'avg_monthly_bill_dlrs_scld', 'pop_under_18_scld',
                'education_master_scld', 'education_doctoral_scld']

# good acc: 0.7436464088397791, R2 0.1561717889129457
good_acc_lr3 = ['Adoption', 'land_area_scld', 'education_bachelor_scld', 'travel_time_average_scld', 'hh_size_4_scld',
                'education_master_rate', 'mod_sf_own_mwh_scld', 'number_of_years_of_education_scld',
                'bachelor_or_above_rate', 'hu_own_scld', 'hu_vintage_2000to2009_scld', 'education_bachelor_rate']

# good acc: 0.7152854511970534, R2 0.12101327282984775
good_acc_lr4 = ['education_bachelor_scld', 'total_area_scld', 'hu_vintage_1939toearlier_scld', 'heating_fuel_other_scld']

# good acc: 0.7296500920810313, R2 0.18002674246895034
good_acc_lr5 = ['education_bachelor_scld', 'travel_time_average_scld', 'heating_fuel_housing_unit_count_scld',
                'bachelor_or_above_rate', 'land_area_scld', 'pop_female_scld', 'population_density_scld',
                'hh_size_2_scld', 'net_metering_hu_own', 'pop_over_65_scld', 'pop_total_scld',
                'housing_unit_median_gross_rent_scld', 'med_inc_ebill_dlrs_scld', 'hu_own_scld']

# good acc: 0.7340699815837938, R2 0.1713015497234639
good_acc_lr6 = ['land_area_scld', 'pop25_some_college_plus_scld', 'travel_time_average_scld',
                'population_density_scld', 'heating_fuel_coal_coke_scld', 'fam_med_income_scld',
                'heating_fuel_housing_unit_count_scld', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
                'pop_over_65_scld']

# good acc: 0.7355432780847145, R2 0.18448226867693285
good_acc_lr7 = ['education_bachelor_scld', 'land_area_scld', 'pop_male_scld', 'hu_vintage_1939toearlier_scld',
                'population_density_scld', 'education_doctoral_scld', 'travel_time_average_scld',
                'bachelor_or_above_rate', 'pop_over_65_scld', 'heating_fuel_coal_coke_scld',
                'avg_inc_ebill_dlrs_scld', 'E_AGE17_scld']


good_acc_Aavg = mean([.7178637200736648, 0.7436464088397791, .7152854511970534, 0.7296500920810313, 0.7340699815837938,
                      0.7355432780847145])
good_acc_Ravg = mean([0.1561717889129457, 0.12101327282984775, 0.18002674246895034, 0.1713015497234639, 0.18448226867693285])
good_acc_votes = [good_acc_lr, good_acc_lr2, good_acc_lr3, good_acc_lr4, good_acc_lr5, good_acc_lr6]

# good Rsqr: 0.20098292514467608, Accuracy 0.7152854511970534
good_Rsqr4 = ['Adoption', 'pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'heating_fuel_coal_coke_scld',
              'population_density_scld', 'hu_med_val_scld', 'pop_total_scld', 'hu_vintage_1939toearlier_scld',
              'education_bachelor_rate', 'incent_cnt_res_own', 'pop_over_65_scld', 'med_inc_ebill_dlrs_scld',
              'heating_fuel_gas_scld', 'pop_under_18_scld', 'net_metering_hu_own', 'education_bachelor_scld',
              'education_master_scld', 'number_of_years_of_education_scld', 'pop25_no_high_school_scld',
              'E_AGE17_scld', 'education_professional_school_scld', 'own_popden_scld', 'hh_size_2_scld',
              'hu_vintage_1980to1999_scld', 'hu_vintage_1960to1970_scld', 'hu_2000toafter_pct_scld',
              'median_household_income_scld', 'avg_monthly_bill_dlrs_scld', 'high_school_or_below_rate',
              'education_population_scld', 'bachelor_or_above_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
              'age_median_scld', 'travel_time_49_89_rate', 'heating_fuel_other_scld', 'hu_vintage_2010toafter_scld',
              'housing_unit_median_gross_rent_scld', 'hu_own_scld', 'hh_size_4_scld', 'total_area_scld',
              'pop_female_scld', 'poverty_family_count_scld', 'hh_size_3_scld', 'heating_fuel_electricity_scld',
              'hu_2000toafter', 'average_household_income_scld', 'avg_inc_ebill_dlrs_scld', 'education_master_rate',
              'education_doctoral_scld', 'age_25_44_rate_scld', 'fam_med_income_scld', 'mod_sf_own_mwh_scld',
              'housing_unit_count_scld', 'education_high_school_graduate_rate', 'heating_fuel_housing_unit_count_scld',
              'hu_own_pct', 'household_count_scld']

# Rsqaure = 0.20090205587065024
good_Rsqr = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld',
             'heating_fuel_coal_coke_scld', 'population_density_scld', 'hu_med_val_scld', 'pop_total_scld',
             'hu_vintage_1939toearlier_scld', 'education_bachelor_rate', 'incent_cnt_res_own', 'pop_under_18_scld',
             'avg_inc_ebill_dlrs_scld', 'housing_unit_median_gross_rent_scld', 'pop_over_65_scld',
             'education_master_scld', 'age_median_scld', 'heating_fuel_gas_scld', 'net_metering_hu_own',
             'average_household_income_scld', 'avg_monthly_bill_dlrs_scld', 'E_AGE17_scld', 'travel_time_49_89_rate',
             'hu_monthly_owner_costs_greaterthan_1000dlrs_scld', 'education_bachelor_scld', 'education_master_rate',
             'age_25_44_rate_scld', 'own_popden_scld', 'hu_vintage_1960to1970_scld', 'hh_size_2_scld',
             'hu_vintage_1980to1999_scld', 'hu_vintage_2000to2009_scld', 'housing_unit_count_scld', 'total_area_scld',
             'education_high_school_graduate_rate', 'pop25_no_high_school_scld', 'education_population_scld',
             'high_school_or_below_rate', 'education_professional_school_scld', 'hh_size_4_scld', 'mod_sf_own_mwh_scld',
             'number_of_years_of_education_scld', 'bachelor_or_above_rate', 'heating_fuel_electricity_scld',
             'median_household_income_scld', 'med_inc_ebill_dlrs_scld', 'poverty_family_count_scld', 'hh_size_3_scld',
             'education_doctoral_scld', 'heating_fuel_housing_unit_count_scld', 'hu_2000toafter', 'pop_female_scld',
             'hu_own_scld', 'hu_own_pct', 'heating_fuel_other_scld', 'hu_2000toafter_pct_scld', 'fam_med_income_scld',
             'masters_or_above_rate']

# good Rsqr: 0.20355407810942183
good_Rsqr2 = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'heating_fuel_coal_coke_scld',
              'population_density_scld', 'hu_med_val_scld', 'pop_total_scld', 'education_bachelor_rate',
              'pop_under_18_scld', 'hu_vintage_1939toearlier_scld', 'incent_cnt_res_own', 'pop_over_65_scld',
              'education_high_school_graduate_rate', 'hu_2000toafter_pct_scld', 'hu_vintage_1960to1970_scld',
              'hu_monthly_owner_costs_greaterthan_1000dlrs_scld', 'avg_inc_ebill_dlrs_scld',
              'housing_unit_median_gross_rent_scld', 'hu_vintage_1980to1999_scld', 'hh_size_2_scld',
              'net_metering_hu_own', 'education_professional_school_scld', 'E_AGE17_scld', 'education_bachelor_scld',
              'high_school_or_below_rate', 'pop25_no_high_school_scld', 'education_population_scld', 'age_median_scld',
              'heating_fuel_electricity_scld', 'pop_male_scld', 'travel_time_49_89_rate', 'mod_sf_own_mwh_scld',
              'education_college_scld', 'number_of_years_of_education_scld', 'med_inc_ebill_dlrs_scld',
              'fam_med_income_scld', 'masters_or_above_rate', 'education_master_rate', 'housing_unit_count_scld',
              'hu_own_pct', 'hh_size_4_scld', 'avg_monthly_bill_dlrs_scld', 'average_household_income_scld',
              'education_doctoral_scld', 'median_household_income_scld', 'poverty_family_count_scld', 'hu_2000toafter',
              'own_popden_scld', 'hu_own_scld', 'hh_size_3_scld', 'total_area_scld', 'age_25_44_rate_scld',
              'hu_vintage_2000to2009_scld', 'heating_fuel_other_scld', 'heating_fuel_gas_scld',
              'heating_fuel_housing_unit_count_scld']

# good Rsqr: 0.1929803665011074, Accuracy 0.7252302025782689
good_Rsqr3 = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'heating_fuel_coal_coke_scld',
              'population_density_scld', 'hu_med_val_scld', 'pop_total_scld', 'education_bachelor_rate',
              'pop_over_65_scld', 'hu_vintage_1939toearlier_scld', 'incent_cnt_res_own', 'med_inc_ebill_dlrs_scld',
              'heating_fuel_gas_scld', 'education_bachelor_scld', 'housing_unit_median_gross_rent_scld',
              'pop_under_18_scld', 'E_AGE17_scld', 'hh_size_2_scld', 'median_household_income_scld',
              'net_metering_hu_own', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld', 'hu_own_scld',
              'heating_fuel_housing_unit_count_scld', 'housing_unit_count_scld', 'avg_monthly_bill_dlrs_scld',
              'number_of_years_of_education_scld', 'pop25_no_high_school_scld', 'bachelor_or_above_rate',
              'education_professional_school_scld', 'high_school_or_below_rate', 'education_population_scld',
              'hu_own_pct', 'age_median_scld', 'hu_vintage_1960to1970_scld', 'hu_2000toafter_pct_scld',
              'hu_vintage_1980to1999_scld', 'hu_vintage_2010toafter_scld', 'poverty_family_count_scld',
              'hh_size_1_scld', 'age_25_44_rate_scld', 'education_doctoral_rate', 'education_high_school_graduate_rate',
              'education_doctoral_scld', 'hu_2000toafter', 'heating_fuel_electricity_scld', 'travel_time_49_89_rate',
              'hh_size_3_scld', 'fam_med_income_scld', 'pop_female_scld', 'mod_sf_own_mwh_scld', 'heating_fuel_other_scld',
              'total_area_scld', 'own_popden_scld', 'education_master_scld', 'average_household_income_scld',
              'avg_inc_ebill_dlrs_scld']

# good Rsqr: 0.18201814141868777, Accuracy 0.7344383057090239
good_Rsqr5 = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'population_density_scld',
              'heating_fuel_coal_coke_scld', 'hu_med_val_scld', 'pop_total_scld', 'hu_vintage_1939toearlier_scld',
              'education_bachelor_rate', 'incent_cnt_res_own', 'pop_over_65_scld', 'pop_under_18_scld', 'E_AGE17_scld']

LR_ac = ['education_bachelor_scld',
'land_area_scld',
'travel_time_average_scld',
'total_area_scld',
'pop_total_scld',
'population_density_scld',
'hu_vintage_1939toearlier_scld',
'net_metering_hu_own',
'heating_fuel_housing_unit_count_scld',]

# good Rsqr: 0.18966013446910612, Accuracy 0.7233885819521179
good_Rsqr6 = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'population_density_scld',
              'heating_fuel_coal_coke_scld', 'hu_med_val_scld', 'pop_total_scld', 'education_bachelor_rate',
              'hu_vintage_1939toearlier_scld', 'incent_cnt_res_own', 'pop_over_65_scld']

Rsqr_LRs = ['pop25_some_college_plus_scld',
     'travel_time_average_scld',
    'land_area_scld',
    'heating_fuel_coal_coke_scld',
    'population_density_scld',
    'hu_med_val_scld',
    'pop_total_scld',
    'hu_vintage_1939toearlier_scld',
    'education_bachelor_rate',
    'incent_cnt_res_own',]

Rsqr_csn = ['pop25_some_college_plus_scld',
            'travel_time_average_scld',
            'land_area_scld',
            'heating_fuel_coal_coke_scld',
            'population_density_scld',
            'hu_med_val_scld',
            'pop_total_scld',
            'hu_vintage_1939toearlier_scld',
            'education_bachelor_rate',
            'incent_cnt_res_own',
            ]



good_R2_Ravg = mean([0.20098292514467608, 0.20090205587065024, 0.20355407810942183, 0.1929803665011074,
                     0.18201814141868777, .18966013446910612])
good_R2_Aavg = mean([0.7252302025782689, 0.7152854511970534, 0.7344383057090239, 0.7233885819521179])
good_R2_votes = [good_Rsqr, good_Rsqr2, good_Rsqr3, good_Rsqr4, good_Rsqr5, good_Rsqr6]


least_model = ['population_density', 'education_bachelor', 'education_high_school_graduate_rate', 'hu_own',
              'masters_or_above_rate','education_population_scld','average_household_income',
              ]

# Estrella R-Squared : 0.17988423234112094
top_ten_predictors = ['bachelor_or_above_rate', 'net_metering', 'incent_cnt_res_own', 'hu_2000toafter',
                      'education_bachelor_scld', 'education_master_scld', 'education_professional_school_scld',
                      'land_area_scld', 'total_area_scld', 'pop25_some_college_plus_scld']

"""           Complete model file paths"""
model_dec_30 = '__Data/__Mixed_models/December/DeepSolar_Model_2019-12-30_mega.xlsx'
model_dec_30_scld = '__Data/__Mixed_models/December/DS_1_12_scld.xlsx'
model_feb_20 = r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__DeepSolar\Feb\Mixed\DeepSolar_Model_Feb2020-02-02-01-16-39.xlsx'


incentives_M = ['Adoption', 'incent_cnt_res_own', 'net_metering_hu_own', 'incentive_count_nonresidential',
                'incentive_count_residential', 'incentive_nonresidential_state_level',
                'incentive_residential_state_level', 'net_metering', 'property_tax_bin']

policy_N = incentives_M + ['Ren', 'dlrs_kwh', 'avg_electricity_retail_rate_scld', 'avg_electricity_retail_rate']

model_files = {'model_dec_30':model_dec_30,
               '':''}

#occu_trunc = load_tree_trunc_features(dffile='__Data/__Mixed_models/occu/occuold/RF_FI_occu_1_5_trunc.xlsx', limit=.08)
#climate_trunc = load_tree_trunc_features(dffile='__Data/__Mixed_models/climate/climateold/RF_FI_climate_1_2_trunc.xlsx', limit=.09)

drops = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax',
                   'solar_system_count', 'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  'solar_system_count_residential',]

dropsPVa = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax', 'solar_panel_area_per_capita_scld', 'solar_system_count_residential_scld',
                  'solar_panel_area_divided_by_area_scld', 'solar_system_count_scld',
                   'solar_system_count', 'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  'solar_system_count_residential',]

dropsPVar = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax', 'solar_panel_area_per_capita_scld', 'solar_system_count_residential_scld',
                  'solar_panel_area_divided_by_area_scld', 'solar_system_count_scld',
                   'solar_system_count', 'solar_panel_area_per_capita',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  'solar_system_count_residential', 'Adoption']
dropsPVres = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax', 'solar_panel_area_per_capita_scld', 'solar_system_count_residential_scld',
                  'solar_panel_area_divided_by_area_scld', 'solar_system_count_scld', 'Adoption',
                   'solar_system_count', 'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  ]

dropsPVcap = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax', 'solar_panel_area_per_capita_scld', 'solar_system_count_residential_scld',
                  'solar_panel_area_divided_by_area_scld', 'solar_system_count_scld', 'Adoption'
                   'solar_system_count', 'solar_panel_area_divided_by_area',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  'solar_system_count_residential',]

RFR_solar_system_count_residential = [
            'solar_system_count_residential',
            'incent_cnt_res_own',
            'daily_solar_radiation',
            'hu_monthly_owner_costs_greaterthan_1000dlrs',
            'dlrs_kwh',
            'net_metering',
            'net_metering_hu_own',
            'land_area',
            'avg_monthly_consumption_kwh',
            'hu_2000toafter',
            'poverty_family_count',
            'heating_design_temperature',
            'population_density_scld',
            ]

dropsBo = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'number_of_solar_system_per_household',
                  # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax',
                   'solar_system_count', 'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
                  #'daily_solar_radiation', 'solar_system_count_residential',]
                  ]



# allows number of solar stuff to be left in for a heat map
drops_Minus_Solar = ['locale_recode', 'climate_zone', 'company_ty', 'eia_id',
                     #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt',
                     # 'FIPS', 'property_tax', 'number_of_solar_system_per_household']
                  'FIPS', 'property_tax']
model_Dec28 = ['Adoption','own_popden_scld', 'cdd_std_scld', 'Green_Travelers', 'total_area_scld', 'masters_or_above_rate',
                '%hh_size_3', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld', '%female', 'hu_1959toearlier_scld', 'locale_dummy',
                'travel_time_49_89_rate', 'diversity', 'age_25_34_rate', 'employ_rate_scld', 'Pro_Occup', 'net_metering_hu_own',
                'average_household_income_scld', 'avg_monthly_consumption_kwh_scld', 'avg_monthly_consumption_kwh_scld',
                'dlrs_kwh', 'avg_monthly_bill_dlrs_scld', 'Ren', 'age_45_54_rate', 'incentive_residential_state_level',
                'number_of_years_of_education_scld', 'net_metering_hu_own', 'education_bachelor_scld', 'incent_cnt_res_own']

model_31 = ['Adoption', 'population_density', 'travel_time_49_89_rate', 'age_10_14_rate', 'age_15_17_rate',
            'occupation_transportation_rate', 'occupation_arts_rate', 'occupation_finance_rate', 'education_master',
            'occupation_construction_rate', 'age_25_34_rate', 'age_45_54_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
            '%hh_size_3', 'diversity', '%hh_size_2', '%hh_size_2', 'Green_Travelers', '%hh_size_1', 'incent_cnt_res_own',
            'Pro_Occup', 'mortgage_with_rate', 'hu_2000toafter_pct', 'cdd_std_scld', '%female', '%male', 'masters_or_above_rate',
            'high_school_or_below_rate', 'net_metering_hu_own', 'total_area', 'dlrs_kwh', 'avg_inc_ebill_dlrs_scld',
            'average_household_income_scld', 'Ren', 'locale_dummy', ]

policy_mixed = ['net_metering', 'property_tax', 'incent_cnt_res_own', 'incentive_count_residential',
                '','','','',
                '', '', '', '']

model_slim = ['Adoption', 'population_density', 'high_school_or_below_rate', 'travel_time_10_19_rate',
              'travel_time_49_89_rate', 'age_65_74_rate', 'age_18_24_rate', 'travel_time_10_19_rate', 'net_metering_hu_own',
              'diversity', 'Green_Travelers', 'hu_1960to1979_pct', 'education_bachelor_scld']

Xu_Modelb = ['Adoption', 'incentive_count_residential_scld', 'incentive_residential_state_level_scld', 'net_meter_cate',
            'dlrs_kwh', 'number_of_years_of_education_scld', 'education_less_than_high_school_rate', 'education_master_rate',
            'median_household_income', 'employ_rate', 'female_pct', 'voting_2012_dem_percentage',
            'hu_own_pct', 'diversity', 'age_35_44_rate', 'age_45_54_rate',
            'age_55_64_rate', 'age_65_74_rate', 'population_density_scld', 'housing_unit_count',
            '%hh_size_3', 'land_area', 'locale_recode', 'hdd',
            'heating_fuel_electricity_rate', 'heating_fuel_coal_coke_rate', 'hu_vintage_2010toafter', 'hu_vintage_1939toearlier',
            'hu_vintage_1940to1959', 'hu_vintage_1960to1970', 'hu_vintage_1980to1999', 'Green_Travelers',
            'avg_monthly_consumption_kwh', 'travel_time_40_59_rate', 'travel_time_60_89_rate']


Xu_ModelC = [
                'incentive_count_residential', 'net_meter_bin', 'Ren',
                'dlrs_kwh',
                'property_tax_bin',
                'number_of_years_of_education',
                'education_HSorBELOW_rate', # TODO: fix this one
                'education_master_or_above_rate',
                'Zmedian_household_income',
                'employ_rate',
                'female_pct',
                'voting_2012_dem_percentage',
                'hu_own_pct',
                'diversity',
                'age_55_or_more_rate',
                'population_density_scld',
                'housing_unit_count_scld',
                '%hh_size_4',
                'land_area_scld',
                'locale_recodeRural',               # TODO:  need to create seperate variables for these 3
                'locale_recodeSuburban',
                'locale_recodeTown',
                'hdd_scld',
                'heating_fuel_electricity_rate',
                'heating_fuel_coal_coke_rate',
                'hu_1959toearlier_pct',
                'hu_2000toafter_pct',
                'Green_Travelers',
                'avg_monthly_bill_dlrs'                
                'travel_time_40_89_rate',
                'net_meter_binary1:hu_own_pct',
                'property_tax_binary1:hu_own_pct',
                ]



model_1_20 = ['Adoption', 'population_density_scld', 'education_bachelor_scld', 'travel_time_49_89_rate',
              'occupation_transportation_rate', 'education_doctoral_scld', '%female', 'Green_Travelers',
              'pop25_some_college_plus_scld', 'education_master_scld', '%hh_size_2', 'travel_time_10_19_rate',
              'age_10_14_rate', 'age_more_than_85_rate', 'land_area_scld', 'travel_time_less_than_10_rate',
              'occupation_finance_rate', 'occupation_construction_rate', 'diversity', 'cdd_std_scld',
              'own_popden_scld', 'total_area_scld', 'age_18_24_rate', 'occupation_administrative_rate',
              'heating_fuel_coal_coke_rate', 'incent_cnt_res_own', 'very_low_sf_own_mwh_scld', 'education_high_school_graduate_rate',
              'hu__1980to1999_pct', 'Pro_Occup', 'travel_time_30_39_rate', 'age_65_74_rate',
              'education_college_rate', '%hh_size_3', 'poverty_family_below_poverty_level_rate', 'hu_monthly_owner_costs_lessthan_1000dlrs_scld',
              'hu_2000toafter_pct', '%hh_size_4', 'average_household_size_scld', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
              'household_type_family_rate', 'net_metering_hu_own', 'number_of_years_of_education_scld', 'avg_monthly_consumption_kwh_scld',
              'avg_inc_ebill_dlrs_scld', 'voting_2012_gop_percentage', 'voting_2012_dem_percentage', 'Ren',
              'dlrs_kwh', 'bachelor_or_above_rate', 'incentive_count_residential', 'locale_dummy']

income_stat = [ 'average_household_income',
                'average_household_income_scld',
                'fam_med_income',
                'fam_med_income_scld',
                'median_household_income',
                'median_household_income_scld',
              ]

edu_1_1 =  ['Adoption', 'number_of_years_of_education',
            'education_less_than_high_school_rate','masters_or_above_rate', 'bachelor_or_above_rate',
            'high_school_or_below_rate',
            'education_doctoral', 'education_doctoral_scld','education_doctoral_rate',
            'education_master', 'education_master_scld', 'education_master_rate',
            'education_bachelor', 'education_bachelor_scld','education_bachelor_rate',
            'education_college', 'education_college_rate', 'education_college_scld',
            'education_high_school_graduate', 'education_high_school_graduate_rate', 'education_high_school_graduate_scld',
            'education_less_than_high_school','education_less_than_high_school_rate', 'education_less_than_high_school_scld',
            'education_professional_school','education_professional_school_scld',
            'education_professional_school_rate',
            ]
edu_1_8 = ['Adoption',
           'education_bachelor',
           'education_bachelor_rate',
           'education_bachelor_scld',
           'education_college',
           'education_college_rate',
           'education_college_scld',
           'education_doctoral',
           'education_doctoral_rate',
           'education_doctoral_scld',
           'education_high_school_graduate',
           'education_high_school_graduate_rate',
           'education_high_school_graduate_scld',
           'education_less_than_high_school',
           'education_less_than_high_school_rate',
           'education_less_than_high_school_scld',
           'education_master',
           'education_master_rate',
           'education_master_scld',
           'masters_or_above_rate',
           'education_professional_school',
           'education_professional_school_rate',
           'education_professional_school_scld',
           'bachelor_or_above_rate'
           ]

gender_stat = [
                'pop_female',
                'pop_female_scld',
                'pop_male',
                'pop_male_scld',
                '%female',
                '%male',
              ]

housing_1_8 = [
              'hu_vintage_1939toearlier',
              'hu_vintage_1939toearlier_scld',
              'hu_vintage_1940to1959',
              'hu_vintage_1940to1959_scld',
              'hu_vintage_1960to1970',
              'hu_vintage_1960to1970_scld',
              'hu_vintage_1980to1999',
              'hu_vintage_1980to1999_scld',
              'hu_vintage_2000to2009',
              'hu_vintage_2000to2009_scld',
              'hu_vintage_2010toafter',
              'hu_vintage_2010toafter_scld',
              'housing_unit_median_value',
              'housing_unit_median_value_scld',
              'hu_1959toearlier',
              'hu_1959toearlier_scld',
              'hu_1960to1979_pct',
              'hu_2000toafter',
              'hu_2000toafter_pct',
              'hu_2000toafter_scld',
              'hu__1980to1999_pct',
              'hu_med_val',
              'hu_med_val_scld',
              ]

age_1_B = [
           'age_10_14_rate',
           'age_15_17_rate',
           'age_18_24_rate',
           'age_25_34_rate',
           'age_35_44_rate',
           'age_45_54_rate',
           'age_55_64_rate',
           'age_5_9_rate',
           'age_65_74_rate',
           'age_75_84_rate',
           'age_median',
           'age_median_scld',
           'age_more_than_85_rate',
           'E_AGE17',
           'E_AGE17_scld',
           'education_population',
           'education_population_scld',
           'pop_med_age',
           'pop_med_age_scld',
           'pop_over_65',
           'pop_over_65_scld',
           'pop_under_18',
           'pop_under_18_scld'
       ]

age_1_7 = [
           'age_10_14_rate',
           'age_15_17_rate',
           'age_18_24_rate',
           'age_25_34_rate',
           'age_35_44_rate',
           'age_45_54_rate',
           'age_55_64_rate',
           'age_5_9_rate',
           'age_65_74_rate',
           'age_75_84_rate',
           'age_median',
           'age_median_scld',
           'age_more_than_85_rate',
           'E_AGE17',
           'education_population_scld',
           'pop_med_age',
           'pop_med_age_scld',
           'pop_over_65',
           'pop_over_65_scld',
           'pop_under_18',
           'pop_under_18_scld'
       ]


fam_stat = [
            '%hh_size_1',           # houshold size
            '%hh_size_2',
            '%hh_size_3',
            '%hh_size_4',
            'average_household_size',
            'average_household_size_scld',
            'hh_size_1',
            'hh_size_1_scld',
            'hh_size_2',
            'hh_size_2_scld',
            'hh_size_3',
            'hh_size_3_scld',
            'hh_size_4',
            'hh_size_4_scld',
            'hh_total',
            'hh_total_scld',
            'household_count',
            'household_count_scld',
            'household_type_family_rate',
            'poverty_family_below_poverty_level',
            'poverty_family_below_poverty_level_rate',
            'poverty_family_below_poverty_level_scld',
            'poverty_family_count',
            'poverty_family_count_scld',
            'hu_own',
            'hu_own_pct',
            'hu_own_scld',
            ]


demo_1_2 = ['Adoption', 'number_of_years_of_education', 'education_less_than_high_school_rate',
            'education_bachelor_scld', 'education_less_than_high_school_rate', 'education_high_school_graduate_rate',
            'education_bachelor', 'education_bachelor_rate', 'education_master_scld', 'education_master_rate',
            'education_doctoral_rate', 'masters_or_above_rate', 'bachelor_or_above_rate', 'high_school_or_below_rate',
            'education_population', 'age_55_64_rate', 'age_65_74_rate', 'age_75_84_rate', 'age_more_than_85_rate',
            'age_25_34_rate', 'age_median', 'fam_med_income', 'median_household_income', 'average_household_income_scld',
            'average_household_income', 'diversity', 'pop_female', '%female', '%male', 'Anti_Occup', 'Pro_Occup',
            'employ_rate', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'hu_own', 'hu_own_pct',
            'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4', '%hh_size_1', '%hh_size_2', '%hh_size_4',
            'education_population_scld', 'hh_total', 'employ_rate', '%hh_size_2',
            'high_school_or_below_rate', 'average_household_size', 'average_household_size_scld']
demo_1_2 = list(set(demo_1_2))

demo_top_00 = pd.read_excel('__Data/__Mixed_models/demo/demoold/top_00_vars_demoAra.xlsx', usecols=['Variable'])['Variable'].values.tolist() + ['Adoption']
demo_top_00_nopop = pd.read_excel('__Data/__Mixed_models/demo/demoold/top_00_vars_demoa_nopop.xlsx', usecols=['Variable'])['Variable'].values.tolist() + ['Adoption']
#full_model_1_9 = pd.read_excel('__Data/__Mixed_models/mixed/full_model_12_28.xlsx')
cost = [
         'Adoption',
              'dlrs_kwh',
              'avg_electricity_retail_rate',
              'avg_electricity_retail_rate_scld',
              'housing_unit_median_gross_rent',
              'housing_unit_median_gross_rent_scld',
              'hu_monthly_owner_costs_greaterthan_1000dlrs',
              'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
              'hu_monthly_owner_costs_lessthan_1000dlrs',
              'hu_monthly_owner_costs_lessthan_1000dlrs_scld',
          ]

cost = [
         'Adoption',
              'dlrs_kwh',
              'avg_electricity_retail_rate',
              'avg_electricity_retail_rate_scld',
              'housing_unit_median_gross_rent',
              'housing_unit_median_gross_rent_scld',
              'hu_monthly_owner_costs_greaterthan_1000dlrs',
              'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
              'hu_monthly_owner_costs_lessthan_1000dlrs',
              'hu_monthly_owner_costs_lessthan_1000dlrs_scld',
          ]

full_model_1_9 = pd.read_excel('__Data/__Mixed_models/mixed/full_model_12_28.xlsx', usecols=['Variable'])['Variable'].values.tolist() + ['Adoption']

# Score: 0.7297246212388425, Good is now:
top_3 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education']

heating = [
             'Adoption',
             'heating_fuel_coal_coke',
             'heating_fuel_coal_coke_rate',
             'heating_fuel_coal_coke_scld',
             'heating_fuel_electricity',
             'heating_fuel_electricity_rate',
             'heating_fuel_electricity_scld',
             'heating_fuel_fuel_oil_kerosene',
             'heating_fuel_fuel_oil_kerosene_rate',
             'heating_fuel_fuel_oil_kerosene_scld',
             'heating_fuel_gas',
             'heating_fuel_gas_rate',
             'heating_fuel_gas_scld',
             'heating_fuel_housing_unit_count',
             'heating_fuel_housing_unit_count_scld',
             'heating_fuel_none',
             'heating_fuel_none_rate',
             'heating_fuel_none_scld',
             'heating_fuel_other',
             'heating_fuel_other_rate',
             'heating_fuel_other_scld',
             'heating_fuel_solar',
             'heating_fuel_solar_rate',
             'heating_fuel_solar_scld',
            ]


forward_sel  = ['population_density_scld', 'education_bachelor_scld', 'E_DAYPOP', 'pop_under_18',
                'cdd_std_scld', 'diversity', 'own_popden_scld', 'cooling_design_temperature_scld',
                'education_college_rate', 'property_tax_bin', 'cooling_design_temperature', 'cdd_std',
                'incentive_count_residential']

pop_1_8 = [
              'Adoption',
              'pop_total',
              'pop_total_scld',
              'pop_under_18',
              'pop_under_18_scld',
              'hh_total',
              'hh_total_scld',
              'E_DAYPOP',
              'E_DAYPOP_scld',
              'population_density',
              'population_density_scld',
              'household_count',
              'household_count_scld',
              'housing_unit_count',
              'housing_unit_count_scld',
           ]

pop_2_1_nonredn = [
                      'Adoption',
                      'pop_total',
                      'pop_under_18',
                      'hh_total',
                      'E_DAYPOP',
                      'population_density',
                      'household_count',
                      'housing_unit_count',
                  ]


pop_1_2 = ['Adoption', 'E_DAYPOP', 'population_density', 'E_DAYPOP_scld', 'population_density_scld',
            'pop_total', 'household_count', 'housing_unit_count',
            ]

habit_1_5 = ['Adoption', 'Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
             'travel_time_40_59_rate', 'travel_time_10_19_rate', 'travel_time_20_29_rate',
             'travel_time_60_89_rate', 'travel_time_49_89_rate', 'transportation_home_rate',
             'travel_time_30_39_rate', 'travel_time_average', 'travel_time_less_than_10_rate',
             'transportation_bicycle_rate', 'transportation_car_alone_rate', 'transportation_carpool_rate',
             'transportation_motorcycle_rate', 'transportation_public_rate', 'transportation_walk_rate',
             ]

policy_1_2 = ['Adoption', '', '', '', '',
               '', '', '', '',
               '', '', '', '']

physical_1_2 = ['Adoption', 'heating_fuel_coal_coke_rate', 'heating_fuel_electricity_rate', 'hu_vintage_1939toearlier', 'hu_vintage_1940to1959',
                'hu_vintage_1960to1970', 'hu_vintage_1980to1999', 'hu_vintage_2000to2009', 'hu_vintage_2010toafter',
                'hu_1959toearlier', 'hu_2000toafter', 'household_count', 'hu_own_pct', 'hu_own',
                'hu_monthly_owner_costs_lessthan_1000dlrs', 'hu_monthly_owner_costs_greaterthan_1000dlrs',
                'hu_med_val', 'hu_med_val_scld', 'hu_mortgage', 'heating_fuel_fuel_oil_kerosene_rate',
                'housing_unit_count_scld', 'housing_unit_count',  'hu_vintage_2000to2009_scld',
                'hu_1959toearlier_scld', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
                'hu_monthly_owner_costs_lessthan_1000dlrs_scld']
physical_trunc = load_tree_trunc_features(dffile='__Data/__Mixed_models/physical/phyold/RF_FI_physical_trunc_2200 _20a.xlsx', limit=.03)

geo_1_5 = ['Adoption', 'land_area', 'locale_dummy', 'total_area', ]
geo_1_8 = [
           'Adoption',
           'land_area',
           'land_area_scld',
           'locale_dummy',
           'total_area',
           'total_area_scld',
           ]

e_cost = [
              'dlrs_kwh',
              'avg_electricity_retail_rate',
              'avg_electricity_retail_rate_scld',
         ]

climate_1_2 = ['Adoption', 'cooling_design_temperature', 'cdd', 'heating_design_temperature', 'hdd',
               'cdd_std_scld', 'cdd_std', 'climate_zone', 'cdd_scld', 'hdd_scld',
               'hdd_std', 'hdd_std_scld', 'heating_design_temperature_scld', 'cooling_design_temperature_scld',
               ]

occu_1_5 = ['Adoption', 'occupation_administrative_rate', 'occupation_agriculture_rate', 'occupation_arts_rate',
            'occupation_construction_rate', 'occupation_education_rate', 'occupation_finance_rate',
            'occupation_information_rate', 'occupation_manufacturing_rate', 'occupation_public_rate',
            'occupation_retail_rate', 'occupation_transportation_rate', 'occupation_wholesale_rate',
            ]

ownership = [
             'Adoption',
             'hu_mortgage',
             'hu_mortgage_scld',
             'mortgage_with_rate',
             'hu_own',
             'hu_own_pct',
             'hu_own_scld',
             ]


best_guess = ['Adoption', 'population_density', 'avg_inc_ebill_dlrs', 'pop25_some_college_plus',
              'hu_own', 'education_bachelor', 'masters_or_above_rate', 'diversity', 'incent_cnt_res_own',
              'incentive_count_residential', 'hu_2000toafter',
              ]

mixed_1_2 = ['Adoption', '', '', '', '',
               '', '', '', '',
               '', '', '', '']
mixed_dif1 = pd.read_excel('__Data/__Mixed_models/some_dif_vars.xlsx')['Variable'].values.tolist()

mix_1_12 = ['Adoption','population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'total_area_scld',
            'age_65_74_rate', 'poverty_family_below_poverty_level_rate', 'heating_fuel_solar_scld',
            'travel_time_10_19_rate', 'occupation_finance_rate']

RF_10v = ['E_DAYPOP_scld', 'number_of_years_of_education', 'population_density',
          'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'avg_electricity_retail_rate_scld', 'travel_time_10_19_rate',
          'hu_2000toafter_scld', 'heating_fuel_none', 'pop_over_65', 'incentive_count_nonresidential']

model_empty = ['Adoption', '', '', '', '',
               '', '', '', '',
               '', '', '', '',
               ]

model_dict_blocks = {'population': pop_1_2,
                     'demo':demo_1_2}

acc_sets = [simple_set, simple_set2, simple_set3, simple_set4, simple_set5, simple_set6, simple_set7,
            simple_set8, simple_set9, simple_set9, simple_set10, simple_set11]
Rsqr_sets = [good_Rsqr, good_Rsqr2, good_Rsqr3, good_Rsqr4]
LRacc_sets = [good_acc_lr, good_acc_lr2,good_acc_lr3, good_acc_lr4, good_acc_lr5,]

RFaccuracy_votes = tally_var_votes(acc_sets, )
Rsqr_votes = tally_var_votes(Rsqr_sets,)
LRacc_votes = tally_var_votes(LRacc_sets,)

RF_voted = voted_list(RFaccuracy_votes, thresh=2)
Rsqr_voted = voted_list(Rsqr_votes, thresh=2)
LRacc_voted = voted_list(LRacc_votes, thresh=2)


# can be used to get to the variables in
# a certain model.
model_dir = {'incentives_M':incentives_M,    # OK
             'policy_N':policy_N,            # OK
             'policy_mixed':policy_mixed,    #
             'model_slim':model_slim,        # OK
             'model_Dec28': model_Dec28,     # OK
             'model_31': model_31,           # OK
             'Xu_Modelb': Xu_Modelb,         #
             'model_1_20': model_1_20,       #
             'demo_1_2': demo_1_2,           # OK
             'demo_top_00':demo_top_00,      # OK
             'demo_top_00_nopop':demo_top_00_nopop,      # OK
             'pop_1_2': pop_1_2,             # OK
             'policy_1_2': policy_1_2,       #
             'habbit_1_5': habit_1_5,         #
             'physical_1_2': physical_1_2,   # OK
             'physical_trunc': physical_trunc, # OK
             'geo_1_5': geo_1_5,             # OK
             'climate_1_2': climate_1_2,     # OK
             #'climate_trunc':climate_trunc,  # OK
             'mixed_1_2': mixed_1_2,
             'mixed_dif1':mixed_dif1,
             'occu_1_5':occu_1_5,            # OK
             #'occu_trunc':occu_trunc,        # OK
             'best_guess':best_guess,
             'least_model': least_model,
             'edu_1_1': edu_1_1,
             'age_1_7':age_1_7,
             'edu_1_8':edu_1_8,
             'popdenonly':popdenonly,
             'fam_stat':fam_stat,
             'income_stat':income_stat,
             'gender_stat': gender_stat,
             'housing_1_8':housing_1_8,
             'pop_1_8':pop_1_8,
             'geo_1_8':geo_1_8,
             'cost':cost,
             'e_cost':e_cost,
             'heating':heating,
             'ownership':ownership,
             'full_model_1_9':full_model_1_9,
             'forward_sel': forward_sel,
             'top_ten_predictors':top_ten_predictors,
             'simple_set':simple_set,
             'simple_set2':simple_set2,
             'simple_set3':simple_set3,
             'simple_set4':simple_set4,
             'simple_set5':simple_set5,
             'simple_set6':simple_set6,
             'simple_set7':simple_set7,
             'simple_set8':simple_set8,
             'simple_set9':simple_set9,
             'simple_set10':simple_set10,
             'good_acc_lr':good_acc_lr,
             'good_acc_lr2':good_acc_lr2,
             'good_acc_lr3':good_acc_lr3,
             'good_acc_lr4':good_acc_lr4,
             'good_acc_lr5':good_acc_lr5,
             'good_Rsqr':good_Rsqr,
             'good_Rsqr2':good_Rsqr2,
             'good_Rsqr3':good_Rsqr3,
             'good_Rsqr4':good_Rsqr4,
             'RF_10v':RF_10v,
             'simple_set11':simple_set11,
             'simple_set12': simple_set12,
             'simple_set13': simple_set13,
             'good_acc_lr7':good_acc_lr7,
             'LR_ac':LR_ac,
             'Rsqr_LRs':Rsqr_LRs,
             'RF_voted':RF_voted,
             'Rsqr_voted':Rsqr_voted,
             'LRacc_voted':LRacc_voted,
             }         #

block_directories = {'climate':['__Data/__Mixed_models/climate/'],
                     'demo':['__Data/__Mixed_models/demo/'],
                     'geo':['__Data/__Mixed_models/geo/'],
                     'habit':['__Data/__Mixed_models/habbit/'],
                     'mixed':['__Data/__Mixed_models/mixed/'],
                     'occu':['__Data/__Mixed_models/occu/'],
                     'policy':['__Data/__Mixed_models/policy/'],
                     'physical':['__Data/__Mixed_models/physical/'],
                     'population':['__Data/__Mixed_models/population/'],
                     'education':['__Data/__Mixed_models/education/']
                     }

# names of feature lists to choose
# use the index to load from file strings
possible_features = ['incentives_M',    # OK 0
                     'policy_N',        # OK 1
                     'model_slim',      # OK 2
                     'model_Dec28',     # OK 3
                     'model_31',        # OK 4
                     'Xu_Modelb',       # OK 5
                     'demo_1_2',        # OK 6
                     'climate_1_2',     # OK 7
                     'pop_1_2',         # OK 8
                     'geo_1_5',         # OK 9
                     'habbit_1_5',      # OK 10
                     'physical_1_2',    # OK 11
                     'occu_1_5',        # OK 12
                     'physical_1_2',    # OK 13
                     'mixed',           # OK 14
                     'demo_top_00',       # OK 15
                     'demo_top_00_nopop', # OK 16
                     'occu_trunc',        # OK 17
                     'physical_trunc',    # OK 18
                     'mixed_dif1',      # OK 19
                     'best_guess',      # OK 20
                     'climate_trunc',   # OK 21
                     'least_model',     # 22
                     'edu_1_1',         # 23
                     'age_1_7',         # 24
                     'edu_1_8',         # 25
                     'popdenonly',      # 26
                     'fam_stat',        # 27
                     'gender_stat',     # 28
                     'income_stat',     # 29
                     'pop_1_8',         # 30
                     'geo_1_8',         # 31
                     'housing_1_8',     # 32
                     'cost',            # 33
                     'e_cost',          # 34
                     'heating',         # 35
                     'ownership',       # 36
                     'full_model_1_9',  # 37
                     'forward_sel',     # 38
                     'simple_set',      # 39
                     'simple_set2',     # 40
                     'simple_set3',     # 41
                     'simple_set4',     # 42
                     'simple_set5',     # 43
                     'simple_set6',     # 44
                     'simple_set7',     # 45
                     'good_acc_lr',     # 46
                     'good_acc_lr2',    # 47
                     'good_acc_lr3',    # 48
                     'good_acc_lr4',    # 49
                     'good_acc_lr5',    # 50
                     'simple_set8',     # 51
                     'simple_set9',     # 52
                     'simple_set10',    # 53
                     'good_Rsqr',       # 54
                     'good_Rsqr2',      # 55
                     'good_Rsqr3',      # 56
                     'good_Rsqr4',      # 57
                     'top_ten_predictors', # 58
                     'RF_10v',          # 59
                     'simple_set12',     # 60
                     'simple_set11',     # 61
                     'good_acc_lr7',    #  62
                     'LR_ac',            # 63
                     'Rsqr_LRs',         # 64
                     'RF_voted',         # 65
                     'Rsqr_voted',      # 66
                     'LRacc_voted',      # 67
                     'simple_set13',     # 68
                     ]

file_strings = {possible_features[0]:'incentives',
                # possible_features[]:'',
                possible_features[1]:'policy',
                possible_features[2]:'model_slim',
                possible_features[3]:'model_Dec28',
                possible_features[4]:'model_31',
                possible_features[5]:'Xu_Models',                     # TODO: need to add special case for this one
                possible_features[7]:'climate',
                possible_features[6]:'demo',
                possible_features[9]:'geo',
                possible_features[10]:'habbit',
                possible_features[11]:'incentives',
                possible_features[14]:'mixed',
                possible_features[12]:'occu',
                possible_features[13]:'physical',
                possible_features[8]:'population',
                possible_features[15]:'demo',
                possible_features[16]:'demo',
                possible_features[17]:'occu',
                possible_features[18]:'physical',
                possible_features[19]:'mixed',
                possible_features[20]:'mixed',
                possible_features[21]:'climate',
                possible_features[22]:'mixed',
                possible_features[23]:'education',
                possible_features[24]: 'age',
                possible_features[25]: 'education',
                possible_features[26]: 'mixed',
                possible_features[27]: 'family',
                possible_features[28]: 'gender',
                possible_features[29]: 'income',
                possible_features[30]: 'population',
                possible_features[31]: 'geo',
                possible_features[32]: 'housing',
                possible_features[33]: 'cost',
                possible_features[34]: 'ecost',
                possible_features[35]: 'heating',
                possible_features[36]: 'ownership',
                possible_features[37]: 'mixed',
                possible_features[38]: 'forward_sel',
                possible_features[39]: 'forward_sel',
                possible_features[40]: 'forward_sel',
                possible_features[41]: 'forward_sel',
                possible_features[42]: 'forward_sel',
                possible_features[43]: 'forward_sel',
                possible_features[44]: 'forward_sel',
                possible_features[45]: 'forward_sel',
                possible_features[46]: 'forward_sel',
                possible_features[47]: 'forward_sel',
                possible_features[48]: 'forward_sel',
                possible_features[49]: 'forward_sel',
                possible_features[50]: 'forward_sel',
                possible_features[51]: 'forward_sel',
                possible_features[52]: 'forward_sel',
                possible_features[53]: 'forward_sel',
                possible_features[54]: 'forward_sel',
                possible_features[55]: 'forward_sel',
                possible_features[56]: 'forward_sel',
                possible_features[57]: 'forward_sel',
                possible_features[58]: 'forward_sel',
                possible_features[59]: 'forward_sel',
                possible_features[60]: 'forward_sel',
                possible_features[61]: 'forward_sel',
                possible_features[62]: 'forward_sel',
                possible_features[63]: 'forward_sel',
                possible_features[64]: 'forward_sel',
                possible_features[65]: 'forward_sel',
                possible_features[66]: 'forward_sel',
                possible_features[67]: 'forward_sel',
                }


"""              These are the features you want in the census tract shape files for the heat maps      """
"""              need to have        """
heat_map = ['fips', 'number_of_solar_system_per_household', 'incentive_count_residential',
            'Ren', 'avg_electricity_retail_rate_scld', 'dlrs_kwh', 'net_metering', 'Adoption',
            'incentive_count_nonresidential', 'population_density', 'education_bachelor_scld',
            'incentive_residential_state_level', 'solar_system_count', 'solar_panel_area_divided_by_area',
            'solar_panel_area_per_capita', 'daily_solar_radiation', 'solar_system_count_residential',
            ]


def model_selector(model_number):
    """    This will return a list of the variables for
           specific model, the direcotroy name to same it to
           and the actual data columns called use cols
           The model number corresponds to the index into possible features model dictionary
    """
    model_vars = possible_features[model_number]        # grabs featurs  and adoption
    dir_name = file_strings[model_vars]                 # what diretory to save the results in based on its grouping
    usecols = model_dir[model_vars]                     # the independent variables of the model
    return model_vars, dir_name, usecols                # all model variables, directory , predictors


def model_loader(target='Adoption', usecols=None, model_file=model_dec_30, new_drops=None, ret_full=True,
                 verbose=False, impute=True, shuffle=False, heatmap=True, drops2=None):
    # will load data frames for a training set and possibly a heat map
    import numpy as np
    df_base = None
    if new_drops is not None:
        drops = new_drops
    if drops2 is None:
        drops2 = drops_Minus_Solar
    if usecols is not None:
        df_base = pd.read_excel(model_file, usecols=list(set(usecols + [target]+ heat_map))) # load data set with target
        hm_usecols = list(set(usecols + [target] + heat_map))           # load the data for the heat map
        df_HM = pd.read_excel(model_file,usecols=hm_usecols)
    else:
        # df_base = pd.read_excel(model_dec_30_scld).drop(columns=['Anti_Occup_scld', 'E_DAYPOP_scld', 'E_MINRTY_scld',
        #                                                     'cust_cnt_scld', 'employ_rate_scld', ])
        df_base = pd.read_excel(model_file).drop(columns=drops)
        df_HM = pd.read_excel(model_file).drop(columns=drops2)

    # remove target from variable list for training
    dfattribs = list(df_base.columns.values.tolist()).copy()
    ogdfattribs = list(df_HM.columns.values.tolist()).copy()
    # show_list(list(df_base.columns.values.tolist()))
    del dfattribs[dfattribs.index(target)]
    del ogdfattribs[ogdfattribs.index(target)]

    # clean up the data by removing missing data
    df0 = df_base.loc[:, [target]+ dfattribs]
    dfOG = df_HM.loc[:, [target]+ ogdfattribs]
    if shuffle:
        from _products.utility_fnc import shuffle_deck
        shuffle_deck(df0)
        shuffle_deck(dfOG)
    if impute:
        df0.replace(np.inf, np.nan, inplace=True)
        df0.replace(-999, np.nan, inplace=True)
        df0.replace('', np.nan, inplace=True)
        df0 = df0.dropna()
        dfOG.replace(-999, np.nan, inplace=True)
        dfOG.replace('', np.nan, inplace=True)
        dfOG = dfOG.dropna()
    df0.drop(columns=heat_map)
    if verbose:
        print("The features in the model")
        print(dfattribs)
        print('Model Statistics:')
        print(df0.describe())
    # df0 training data frame, dfog used to make heat map
    return (df0, dfOG), dfattribs, (df0.loc[:, dfattribs], df0[target])


def load_model(target='', usecols=None, model_file=model_feb_20, ret_full=True,
               verbose=False, impute=True, shuffle=False, heatmap=True, pdrops=None):
    # will load data frames for a training set and possibly a heat map
    import numpy as np
    df_base = None
    if pdrops is not None:
        udrops = pdrops
        if usecols is not None:
        #if pdrops is None:
            df_base = pd.read_excel(model_file, usecols=usecols + [target]).drop(columns=udrops) # load data set with target
        else:
            df_base = pd.read_excel(model_file,).drop(columns=udrops) # load data set with target
    else:
        # df_base = pd.read_excel(model_dec_30_scld).drop(columns=['Anti_Occup_scld', 'E_DAYPOP_scld', 'E_MINRTY_scld',
        #                                                     'cust_cnt_scld', 'employ_rate_scld', ])
        # df_base = pd.read_excel(model_file).drop(columns=drops)
        if usecols is None:
            df_base = pd.read_excel(model_file) # load data set with target
        else:
            df_base = pd.read_excel(model_file, usecols=usecols + [target])
            #df_base = pd.read_excel(model_file,).drop(columns=pdrops)

    # remove target from variable list for training
    dfattribs = list(df_base.columns.values.tolist()).copy()
    # show_list(list(df_base.columns.values.tolist()))
    del dfattribs[dfattribs.index(target)]

    # clean up the data by removing missing data
    df0 = df_base.loc[:, [target]+ dfattribs]
    if shuffle:
        from _products.utility_fnc import shuffle_deck
        shuffle_deck(df0)
    if impute:
        df0.replace(-999, np.nan, inplace=True)
        df0.replace('', np.nan, inplace=True)
        df0 = df0.dropna()

    if verbose:
        print("The features in the model")
        print(dfattribs)
        print('Model Statistics:')
        print(df0.describe())

    return df0, dfattribs, (df0.loc[:, dfattribs], df0[target])



class DeepSolarModel:
    def __init__(self, target, model_num=None, model_name=None, model_file=model_dec_30, variables=None,
                 verbose=False, ts=.5):
        self.target, self.model_num, self.model_name, self.model_file, self.variables, self.verbose\
            = target, model_num, model_name, model_file, variables, verbose
        self.model_vars, self.dir_name, self.usecols = model_selector(self.model_num)
        if self.verbose:
            print('Loading model {}'.format(self.model_vars))
            print('Using variables: {}'.format(self.usecols))

        # load the desired model, dropping any needed features
        df0s, Mfeatures,  xy = model_loader(target, self.usecols, self.model_file)
        self.df0 = df0s[0]          # original un altered data set
        self.dfog = df0s[1]         # heat_map_version
        X0 = xy[0].values           # predictors variables
        self.Xdf0 = xy[0]      # heat map x
        self.y0 = xy[1].values      # targets
        self.ydf0 = xy[1]           # heat map targets

        Training, Testing = cross_val_splitter(self.df0, Mfeatures, ts=.5, verbose=True)

        self.X_tr, self.y_tr = Training[0], Training[1]
        self.X_ts, self.y_ts = Testing[0], Testing[1]


"""    Get the usecols or variables we want to use from the data sets"""
def get_DS_NREL_SVI_usecols(paths, ):
    ds = pd.read_excel(paths[0][0], sheet_name=paths[0][1])['variables'].values.tolist()
    nrel = pd.read_excel(paths[1][0], sheet_name= paths[1][1])['variables'].values.tolist()
    svi = pd.read_excel(paths[2][0], sheet_name= paths[2][1])['variables'].values.tolist()
    return ds, nrel, svi

"""          This will select certain states/census tracts ects from the deep solar set"""
def DeepSolarAreaSelector(ds_file, region_type, regions, ds_cols=None):
    if ds_cols is not None:
        ds = pd.read_excel(ds_file, usecols=ds_cols)
    else:
        ds = pd.read_excel(ds_file,)
    ds = ds.loc[ds[region_type].isin(regions)]
    return ds

"""          this will load the data sets for merging      """
def load_the_data(ds_file, region_type, regions, ds_cols,
                  nrel_file, nrel_cols,
                  svi_file, svi_cols, verbose=False):
    ds = DeepSolarAreaSelector(ds_file, region_type, regions, ds_cols)
    nrel = pd.read_excel(nrel_file, usecols=nrel_cols)
    svi = pd.read_excel(svi_file, usecols=svi_cols)
    return ds, nrel, svi

def merge_set_for_processing(set_lists, target, verbose=False):
    from _products.utility_fnc import data_merger
    # dsss = [ds, svi, nrel] == set_lists
    merged = data_merger(set_lists, target=target)
    if verbose:
        print('=======================================')
        print('The merged data set contains: ')
        print(merged.columns.values.tolist())
        print('=======================================')
    return merged

def cross_val_splitter(df0, rl, target='Adoption', ts=.5, verbose=False, stratify=True):
    from sklearn.model_selection import train_test_split
    targets0 = df0[target]
    df0 = df0.loc[:, rl]
    ts = .50
    tr = 1 - ts
    # Create training and testing sets for the data
    if stratify:
        X_train0, X_test0, y_train0, y_test0 = train_test_split(df0, targets0, stratify=targets0, test_size=ts,
                                                                train_size=tr)
    else:
        X_train0, X_test0, y_train0, y_test0 = train_test_split(df0, targets0, test_size=ts,
                                                                train_size=tr)
    if verbose:
        print('Training:')
        print(X_train0.describe())
        print('Testing:')
        print(X_test0.describe())
    return (X_train0, y_train0), (X_test0, y_test0)



def generate_green_travelers(merged, green_travelers=None):
    """ adds a summatin of those whom walk, ride a bike or work from home"""
    from _products.utility_fnc import create_combo_var_sum
    # pass a list of desired variables to sum, and the data frame the come from and get back the result
    if green_travelers is None:
        green_travelers = ['transportation_home_rate', 'transportation_bicycle_rate', 'transportation_walk_rate']
    create_combo_var_sum(merged, green_travelers, newvar='Green_Travelers')
    return

def generate_pro_Anti_occu(merged):
    """     Adds column for a summation of jobs negatively and postively correlated to adoption"""
    from _products.utility_fnc import create_combo_var_sum
    anti_jobs = ['occupation_agriculture_rate', 'occupation_construction_rate', 'occupation_transportation_rate',
                 'occupation_manufacturing_rate']
    pro_jobs = ['occupation_administrative_rate', 'occupation_information_rate', 'occupation_finance_rate',
                'occupation_arts_rate', 'occupation_education_rate']
    create_combo_var_sum(merged, anti_jobs, newvar='Anti_Occup')
    create_combo_var_sum(merged, pro_jobs, newvar='Pro_Occup')
    return

def add_state_Ren(merged, ):
    from _products.utility_fnc import add_renewable_gen
    # ren = {"al":.091, 'ga':.076, 'ky':.062, 'ms':.029, 'nc':.128, 'sc':.053, 'tn':.133, 'va':.059, 'fl':.033}
    """ source for info below: https://www.energy.gov/maps/renewable-energy-production-state """

    # ren = {"al":.1615, 'az':.1201, 'ca':0.2436, 'ga':.3636, 'ma':0.4272, 'ny':0.4479, 'tx':0.255, 'ut':0.151,
    #       'ky':.235, 'ms':.117, 'nc':.2573, 'sc':.1604, 'tn':.3521, 'va':.1054, 'fl':.4099}
    ren = {"al": .1615, 'az': .1201, 'ca': 0.2436, 'ga': .3636, 'ma': 0.4272, 'ny': 0.4479, 'tx': 0.255, 'ut': 0.151,
           'ky': .235, 'ms': .117, 'nc': .2573, 'tn': .3521, 'va': .1054, }
    merged = add_renewable_gen(merged, 'state', ren)
    return

def locale_recode_action(merged, ):
    from _products.utility_fnc import recode_var_sub
    local_recode = {'Rural': 1, 'Town': 2, 'City': 4, 'Suburban': 3, 'Urban': 4}
    # below will replace any string containing the key with the vals
    local_recodeA = {'Rural': 'Rural', 'Town': 'Town', 'City': 'City', 'Suburban': 'Suburban', 'Urban': 'City'}
    sought = ['Rural', 'Town', 'City', 'Suburban', 'Urban']
    local = list(merged['locale'])
    merged['locale_dummy'] = recode_var_sub(sought, local, local_recode)
    merged['locale_recode'] = recode_var_sub(sought, local, local_recodeA)
    return


def gen_edu_combo(merged):
    from _products.utility_fnc import create_combo_var_sum
    high_below = ['education_less_than_high_school_rate', 'education_high_school_graduate_rate']
    merged['high_school_or_below_rate'] = create_combo_var_sum(merged, high_below)

    master_above = ['education_master_rate', 'education_doctoral_rate']
    merged['masters_or_above_rate'] = create_combo_var_sum(merged, master_above)

    bachelor_above = ['education_master_rate', 'education_doctoral_rate'] + ['education_bachelor_rate']
    merged['bachelor_or_above_rate'] = create_combo_var_sum(merged, bachelor_above)

    edu_excludes = ['high_school_or_below_rate', 'masters_or_above_rate',
                    'bachelor_or_above_rate'] + high_below + master_above + bachelor_above

    return edu_excludes


def gen_ownership_pct(merged):
    merged['hu_own_pct'] = (merged['hu_own'] / merged['housing_unit_count']).values.tolist()
    home_excludes = ['hu_own_pct']
    return home_excludes

def net_met_ptx_bin_recode(merged):
    from _products.utility_fnc import thresh_binary_recode
    thresh_binary_recode(merged, 'net_metering', )
    thresh_binary_recode(merged, 'property_tax', )
    policy_excludes = ['net_metering', 'property_tax']
    return policy_excludes

def gen_age_range(merged):
    from _products.utility_fnc import create_combo_var_sum
    # make range from 1959 to earlier variable
    hage1959toearlier = ['hu_vintage_1940to1959', 'hu_vintage_1939toearlier']
    merged['hu_1959toearlier'] = create_combo_var_sum(merged, hage1959toearlier)

    # make 60 to 79 pct variable
    merged['hu_1960to1979_pct'] = (merged['hu_vintage_1960to1970'] / merged['housing_unit_count']).values.tolist()

    # make 80 to 99 pct variable
    merged['hu_1980to1999_pct'] = (merged['hu_vintage_1980to1999'] / merged['housing_unit_count']).values.tolist()

    # make list of variabels to sum to get range variable from 2000 to beyond
    hage2000tobeyond = ['hu_vintage_2000to2009', 'hu_vintage_2010toafter']
    merged['hu_2000toafter'] = create_combo_var_sum(merged, hage2000tobeyond)

    # make percentage variable out of new variable
    merged['hu_2000toafter_pct'] = (merged['hu_2000toafter'] / merged['housing_unit_count']).values.tolist()
    hu_excludes = ['hu_1980to1999_pct', 'hu_2000toafter', 'hu_1960to1979_pct']
    return hu_excludes


def gen_hh_size(merged):
    from _products.utility_fnc import create_combo_var_sum, percentage_generator
    hh_sizes = ['hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4']
    merged['hh_total'] = create_combo_var_sum(merged, hh_sizes)
    merged['%hh_size_1'] = percentage_generator(merged, hh_sizes[0], 'hh_total')
    merged['%hh_size_2'] = percentage_generator(merged, hh_sizes[1], 'hh_total')
    merged['%hh_size_3'] = percentage_generator(merged, hh_sizes[2], 'hh_total')
    merged['%hh_size_4'] = percentage_generator(merged, hh_sizes[3], 'hh_total')


def gender_redodeing(merged):
    from _products.utility_fnc import create_combo_var_sum, percentage_generator
    female_count = 'pop_female'
    male_count = 'pop_male'
    total = 'pop_total'
    create_combo_var_sum(merged, [female_count, male_count], newvar=total)
    percentage_generator(merged, female_count, total, newvar='%female')
    percentage_generator(merged, male_count, total, newvar='%male')
    return

def gen_travel_time_mix(merged):
    from _products.utility_fnc import create_combo_var_sum, percentage_generator
    trav_recodes = ['travel_time_40_59_rate', 'travel_time_60_89_rate']
    create_combo_var_sum(merged, trav_recodes, newvar='travel_time_49_89_rate')
    travel_excludes = ['travel_time_49_89_rate']
    return travel_excludes

def gen_age_ranges(merged):
    from _products.utility_fnc import create_combo_var_sum, percentage_generator
    age_25_44 = ['age_25_34_rate','age_35_44_rate']
    age_25_64 = ['age_25_34_rate','age_35_44_rate', 'age_45_54_rate', 'age_55_64_rate']
    a_25_44 = 'age_25_44_rate'
    a_25_64 = 'age_25_64_rate'
    a_55_more = 'age_55_or_more_rate'
    #merged[a_25_44] = create_combo_var_sum(merged, age_25_44, newvar=a_25_44)
    #merged[a_25_44] = create_combo_var_sum(merged, age_25_44, newvar=a_25_44)
    #merged[a_25_64] = create_combo_var_sum(merged, age_25_64, newvar=a_25_64)
    create_combo_var_sum(merged, age_25_64, newvar=a_55_more)
    create_combo_var_sum(merged, age_25_64, newvar=a_25_64)
    create_combo_var_sum(merged, age_25_64, newvar=a_55_more)
    return

def gen_mixed(merged):
    from _products.utility_fnc import generate_mixed
    net_own = ['net_metering_bin', 'hu_own_pct']
    new_net = 'net_metering_hu_own'
    generate_mixed(merged, net_own, new_net)

    incent_res_own = ['incentive_count_residential', 'hu_own_pct']
    new_incent_own = 'incent_cnt_res_own'
    generate_mixed(merged, incent_res_own, new_incent_own)

    # incent_med_income = ['incentive_residential_state_level', 'median_household_income' ]
    # incent_state_income = 'incent_st_Mincome'
    # generate_mixed(merged, incent_med_income, incent_state_income)

    # incent_avg_income = ['incentive_residential_state_level', 'average_household_income' ]
    # incent_state_Aincome = 'incent_st_Aincome'
    # generate_mixed(merged, incent_avg_income, incent_state_Aincome)

    med_income_ebill = ['avg_monthly_bill_dlrs', 'median_household_income']
    medincebill = 'med_inc_ebill_dlrs'
    generate_mixed(merged, med_income_ebill, medincebill)

    avg_income_ebill = ['avg_monthly_bill_dlrs', 'average_household_income']
    avgincebill = 'avg_inc_ebill_dlrs'
    generate_mixed(merged, avg_income_ebill, avgincebill)

    own_popden = ['population_density', 'hu_own_pct']
    ownpopden = 'own_popden'
    generate_mixed(merged, own_popden, ownpopden)
    mixed_excludes = [new_net, new_incent_own]
    return mixed_excludes

def make_report_files(merged, pearson_fx=None, pearson_fc=None):
    from _products.utility_fnc import today_is
    import numpy as np
    merged = pd.DataFrame(merged.values, dtype=np.float, columns=merged.columns.tolist(), index=merged.index.tolist(), )
    # correlation table TODO: need to set some to pearson instead of kendal's tau
    kencorr = merged.corr(method='kendall').sort_values(by=['Adoption'], ascending=False, inplace=False)
    pearsoncorr = merged.corr(method='pearson').sort_values(by=['Adoption'], ascending=False, inplace=False)

    pearsoncorr.loc[:, 'Adoption'] = kencorr.loc[:, 'Adoption']
    pearsoncorr.loc['Adoption', :] = kencorr.loc['Adoption', :]
    if pearson_fx is None:
        pearson_fx ='__Data/__Mixed_models/December/DeepSolar_Model_correlation_{}_pearson.xlsx'.format(today_is())
        pearson_fc = '__Data/__Mixed_models/December/__DeepSolar_Model_correlation_{}_pearson.csv'.format(today_is())
    pearsoncorr.sort_values(by=['Adoption'], ascending=False, inplace=False).to_excel(pearson_fx)
    merged.corr(method='kendall').sort_values(by=['Adoption'], ascending=False, inplace=False).to_excel('__Data/__Mixed_models/December/DeepSolar_Model_correlation_{}_kendal.xlsx'.format(today_is()))

    pearsoncorr.sort_values(by=['Adoption'], ascending=False, inplace=False).to_csv(pearson_fc)
    merged.corr(method='kendall').sort_values(by=['Adoption'], ascending=False, inplace=False).to_csv('__Data/__Mixed_models/December/__DeepSolar_Model_correlation_{}_kendal.csv'.format(today_is()))
    return

def scale_merged(merged, excludes, scale_sub, gen_scld_only, verbose=True):
    from sklearn.preprocessing import MinMaxScaler
    from _products.utility_fnc import rmv_list_list, today_is
    scaler = MinMaxScaler()
    # list of things to remove
    rmv_scl = list(
        set(pd.read_excel('__Data/__Mixed_models/__Nominal_values_exclude_list.xlsx')['variables'].values.tolist()))
    rmv_scl += excludes
    add_back = list(set(['Adoption'] + excludes ))
    rmv_scl = list(set(rmv_scl))
    # remove the string based or unwanted varibles from set to scale
    #
    ma_tribs = merged.columns.values.tolist()
    scalables = rmv_list_list(ma_tribs, rmv_scl)
    # nrm = NORML()
    scldf = merged.loc[:, scalables]
    if verbose:
        print('remove list', rmv_scl)
        print('merged attribs', ma_tribs)
        print('scalables stuff\n', scldf.columns)
        print(scalables)

    if scale_sub:          # Ot substitute the scaled versions
        # if want to substitute do below
        nscalables = [s + '_scld' for s in scalables]
        # nrm.fit(scldf)
        scldf = pd.DataFrame(scaler.fit_transform(scldf), columns=nscalables, index=merged.index.values.tolist())
        if verbose:
            print('merge shape 1', merged.shape)
            print(nscalables)
            print(scldf)
        if gen_scld_only:
            if verbose:
                print('Only generating a file for scaled variables ')
            scldf.to_excel('__Data/__DeepSolar/Feb/Mixed/DeepSolar_Model_{}_scld_ONLY.xlsx'.format(today_is()),
                           index=False)
            quit(1450)
        merged = merged.join(scldf, lsuffix='', rsuffix='_scld')
        # merged.index = nscalables
        merged.drop(columns=scalables, inplace=True)
        merged = merged.loc[:, add_back + scldf.columns.tolist()]
        print('Scaled data shape: ', merged.shape)
        print(merged)
        # merged.to_excel('__Data/__Mixed_models/December/DS_1_12_scld.xlsx', index=False)
        merged.to_csv('__Data/__DeepSolar/Feb/Mixed/DS_2_2_20_scld.csv', index=False)
        quit(402)
    else:
        # if want to add do below TODO: adding makes it to big create two and do seperate join
        #                               ,look into increasing storage to solve
        print('the data set will contain both version of variables')
        nscalables = [s + '_scld' for s in scalables]
        print(nscalables)
        # nrm.fit(scldf)
        scldf = pd.DataFrame(scaler.fit_transform(scldf), columns=nscalables, index=merged.index.values.tolist())
        print(scldf)
        # scldf.to_excel('_DeepSolar/DeepSolar_Model17Gscld.xlsx', index=False)
        # scaled_merged = nrm.transform(scldf, headers=scldf.columns.values.tolist())
        merged = merged.join(scldf, lsuffix='', rsuffix='_scld')
        print('shape: ', merged.shape)
        print(merged)
        # print(merged)
        return merged

def save_new_data(merged, scale_sub, newname, newreport, ):
    from _products.utility_fnc import today_is, report_var_stats
    if scale_sub:
        mdf = report_var_stats(merged,
                               name=r'__Data/__DeepSolar/Feb/Mixed/DeepSolar_Model_var_stats{}_scld.xlsx'.format(
                                   today_is()))
        # merged.to_excel('__Data/__Mixed_models/December/DeepSolar_Model_{}_scld.xlsx'.format(today_is()), index=False)           # create excel version
        merged.to_csv('__Data/__DeepSolar/Feb/Mixed/DeepSolar_Model_FEB{}_scld.csv'.format(today_is()),
                      index=False)  # create csv version
    else:
        mdf = report_var_stats(merged, name=r'__Data/__Mixed_models/December/DeepSolar_Model_var_stats{}.xlsx'.format(
            today_is()))
        # save the new data set as an excel and csv file
        # merged.to_excel('__Data/__DeepSolar/Feb/Mixed/DeepSolar_Model_{}.xlsx'.format(today_is()), index=False)           # create excel version
        merged.to_csv('__Data/__DeepSolar/Feb/Mixed/DeepSolar_Model_Feb12_scld{}.csv'.format(today_is()),
                      index=False)  # create csv version

class DATA_SETS:
    def __init__(self,):
        self.DS_PATH = r'__Data/__DeepSolar/deepsolar_tract_orig_Adoption.xlsx'
        self.NREL_PATH = r'__Data/__NREL/NREL_seeds.xlsx'
        self.SVI_PATH7 = r'__Data/__SVI/SVI_SE_7.xlsx'
        self.SVI_PATH13 = r'__Data/__SVI/SVI_12.xlsx'
    def get_model(self, model):
        if model.upper() == 'DS':
            return self.DS_PATH
        elif model.upper() == 'NREL':
            return self.NREL_PATH
        elif model.upper() == 'SVI7':
            return self.SVI_PATH7
        elif model.upper() == 'SVI13':
            return self.SVI_PATH13
        else:
            print("Error: unknown data set type {}".format(model))
            print('Calling method get_model(), DATA_SETS class')
            quit(-1576)


model_7st_2_2 = '__Data/____Training/DeepSolar_Model_Feb7_scld.xlsx'


model_12st_init = '__Data/____Training/DeepSolar_Model_Feb13_adopt.xlsx'


