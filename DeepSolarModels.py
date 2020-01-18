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
# ############TODO: Random Forest FS sets  #####################
# ##############################################################
# ##############################################################


# 0.7363917513990309
simple_set = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education',
              'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'avg_electricity_retail_rate_scld', 'heating_fuel_other']

#  0.7377803512494823
simple_set2 = ['population_density_scld', 'E_DAYPOP', 'number_of_years_of_education_scld', 'total_area_scld',
               'age_65_74_rate', 'poverty_family_below_poverty_level_rate', 'heating_fuel_solar_scld',
               'travel_time_10_19_rate', 'occupation_finance_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
               'heating_fuel_coal_coke_scld', 'heating_fuel_none']


#  0.7358360798133626
simple_set3 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'hu_monthly_owner_costs_lessthan_1000dlrs', 'avg_electricity_retail_rate_scld', 'land_area_scld',
               'hu_vintage_1939toearlier_scld']

#  0.7333350960892336, Good is now:
simple_set4 = ['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'pop_over_65', 'hu_2000toafter_scld']

# Score: 0.7384736961325862, Good is now:
simple_set5 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'hu_monthly_owner_costs_lessthan_1000dlrs_scld', 'avg_electricity_retail_rate_scld',
               'incentive_count_nonresidential', 'fam_med_income_scld', 'travel_time_10_19_rate', 'age_75_84_rate']

# Score: 0.7361141471600564, Good is now:
simple_set6 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'age_65_74_rate',
               'total_area', 'heating_fuel_gas_scld', 'hu_2000toafter_scld',
               'hu_monthly_owner_costs_greaterthan_1000dlrs', 'incentive_count_nonresidential']

# Score: 0.7376407100697606
simple_set7 =['population_density', 'E_DAYPOP_scld', 'number_of_years_of_education_scld', 'pop_over_65',
              'hu_2000toafter_scld', 'incent_cnt_res_own', 'travel_time_10_19_rate', 'employ_rate_scld',
              'education_bachelor_rate', 'pop_med_age_scld', 'hh_med_income']

# e: 0.737853762194533, 0.21913339615387883
simple_set8 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
               'avg_electricity_retail_rate_scld', 'travel_time_10_19_rate']

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
# Score: 0.737853762194533, Good is now:
simple_set12 = ['population_density_scld', 'E_DAYPOP_scld', 'number_of_years_of_education',
                'avg_electricity_retail_rate_scld', 'travel_time_10_19_rate']

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
good_acc_lr3 = ['land_area_scld', 'education_bachelor_scld', 'travel_time_average_scld', 'hh_size_4_scld',
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
good_Rsqr4 = ['pop25_some_college_plus_scld', 'travel_time_average_scld', 'land_area_scld', 'heating_fuel_coal_coke_scld',
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


model_dec_30 = '__Data/__Mixed_models/December/DeepSolar_Model_2019-12-30_mega.xlsx'
model_dec_30_scld = '__Data/__Mixed_models/December/DS_1_12_scld.xlsx'



incentives_M = ['Adoption', 'incent_cnt_res_own', 'net_metering_hu_own', 'incentive_count_nonresidential',
                'incentive_count_residential', 'incentive_nonresidential_state_level', 'incentive_residential_state_level',
                'net_metering', 'property_tax_bin']

policy_N = incentives_M + ['Ren', 'dlrs_kwh', 'avg_electricity_retail_rate_scld', 'avg_electricity_retail_rate']

model_files = {'model_dec_30':model_dec_30,
               '':''}

occu_trunc = load_tree_trunc_features(dffile='__Data/__Mixed_models/occu/occuold/RF_FI_occu_1_5_trunc.xlsx', limit=.08)
climate_trunc = load_tree_trunc_features(dffile='__Data/__Mixed_models/climate/climateold/RF_FI_climate_1_2_trunc.xlsx', limit=.09)

drops = ['locale_recode', 'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
                  #'geoid', 'locale', 'number_of_solar_system_per_household_scld',
                  'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'number_of_solar_system_per_household',
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
                'housing_unit_count_scld', 'housing_unit_count', 'hu_2000toafter_scld', 'hu_vintage_2000to2009_scld',
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
             'climate_trunc':climate_trunc,  # OK
             'mixed_1_2': mixed_1_2,
             'mixed_dif1':mixed_dif1,
             'occu_1_5':occu_1_5,            # OK
             'occu_trunc':occu_trunc,        # OK
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
                     'demo_top_00',                      # OK 15
                     'demo_top_00_nopop',                # OK 16
                     'occu_trunc',                # OK 17
                     'physical_trunc',                # OK 18
                     'mixed_dif1',      # OK 19
                     'best_guess',      # OK 20
                     'climate_trunc',    # OK 21
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






