#!/usr/bin/env python3
"""
MAPP Data Separator - De-identified with Proper Labels

This script:
1. Uses metadata to filter variables by event
2. Filters to enrolled participants only 
3. Removes HIPAA identifiers (except subject_id)
4. Creates variable label mappings for descriptive statistics
5. Generates properly labeled output files
"""

import pandas as pd
import os
import json
from datetime import datetime

def clean_for_ascii(df):
    """Clean dataframe for ASCII encoding"""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.encode('ascii', errors='replace').str.decode('ascii')
    return df_clean

def create_form_to_event_mapping():
    """Create mapping from REDCap forms to study events"""
    form_event_mapping = {
        # Enrollment/Screening forms
        'prescreen': ['enrollment_arm_1'],
        'contact_form': ['enrollment_arm_1'], 
        'econsent_form': ['enrollment_arm_1'],
        'study_status_eligibility': ['enrollment_arm_1', 'baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        
        # Baseline forms
        'saliva_collection_log': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'bp': ['baseline_preop_arm_1'],
        'anthropometrics': ['baseline_preop_arm_1'],
        'demographics_health_history': ['baseline_preop_arm_1'],
        'health_history_update': ['week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'body_map': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'use_of_pain_medication_form': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'patient_centered_outcomes': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'hrv_bp': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'copcs': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        
        # PROMIS questionnaires
        'promis_pain_intensity_29fba2': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'promis_pain_interference': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'promis_physical_function': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'adult_situational_hope_scale': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'promis_sf_v10_anxiety_8a': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'promis_depression': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        
        # Physical function tests
        'qst_study_data_sheet': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'pittsburgh_sleep_quality_index_psqi': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'pain_catastrophizing_scale': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'panas': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'life_orientation_test_revised_lotr': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'fear_of_daily_activities': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'pain_selfefficacy': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'perceived_stress_scale_pss10': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'promis_sf_v20_emotional_support_4a': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'sppb': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'timed_up_and_go_test_tug': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        
        # EMA forms  
        'ema_questionnaire': ['baseline_ema_morni_arm_1', 'baseline_ema_after_arm_1', 'baseline_ema_eveni_arm_1',
                             '4_wk_ema_morning_arm_1', '4_wk__ema_afternoo_arm_1', '4_wk__ema_evening_arm_1'],
        
        # Survey triggers and compensation
        'survey_triggers': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'ema_test': ['baseline_preop_arm_1'],
        'subject_reimbursement_form': ['baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1'],
        'contact_notes': ['enrollment_arm_1', 'baseline_preop_arm_1', 'week_4_postop_arm_1', 'month_3_post_op_arm_1']
    }
    
    return form_event_mapping

def load_and_process_metadata(dict_path):
    """Load data dictionary and create mappings"""
    print("Loading data dictionary...")
    data_dict = pd.read_csv(dict_path)
    
    # Create variable to form mapping
    var_to_form = {}
    var_to_label = {}
    var_to_choices = {}
    hipaa_identifiers = set()
    
    for _, row in data_dict.iterrows():
        variable = row['Variable / Field Name'].strip()
        form = row['Form Name'].strip() if pd.notna(row['Form Name']) else 'unknown'
        label = row['Field Label'].strip() if pd.notna(row['Field Label']) else variable
        choices = row['Choices, Calculations, OR Slider Labels'] if pd.notna(row['Choices, Calculations, OR Slider Labels']) else ''
        is_identifier = str(row['Identifier?']).strip().lower() == 'y'
        
        var_to_form[variable] = form
        var_to_label[variable] = label
        var_to_choices[variable] = choices
        
        # Track HIPAA identifiers (but keep subject_id)
        if is_identifier and variable != 'subject_id':
            hipaa_identifiers.add(variable)
    
    print(f"Mapped {len(var_to_form)} variables to forms")
    print(f"Found {len(hipaa_identifiers)} HIPAA identifiers to remove (keeping subject_id)")
    
    return var_to_form, var_to_label, var_to_choices, hipaa_identifiers

def get_variables_for_event(event_name, var_to_form, form_event_mapping):
    """Get list of variables for a specific event"""
    # Always include core REDCap variables (but subject_id will be handled separately for de-identification)
    core_variables = ['subject_id', 'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance']
    
    event_variables = core_variables.copy()
    
    # Add variables from forms used in this event
    for form, events in form_event_mapping.items():
        if event_name in events:
            form_vars = [var for var, var_form in var_to_form.items() if var_form == form]
            event_variables.extend(form_vars)
    
    return list(set(event_variables))

def remove_hipaa_identifiers(df, hipaa_identifiers):
    """Remove HIPAA identifiers except subject_id"""
    columns_to_remove = [col for col in df.columns if col in hipaa_identifiers]
    
    if columns_to_remove:
        print(f"    Removing HIPAA identifiers: {', '.join(columns_to_remove)}")
        df = df.drop(columns=columns_to_remove)
    
    return df

def create_variable_labels_file(var_to_label, var_to_choices, output_dir):
    """Create a comprehensive variable labels file for statistical analysis"""
    labels_data = {}
    
    for variable, label in var_to_label.items():
        labels_data[variable] = {
            'label': label,
            'choices': var_to_choices.get(variable, ''),
            'type': 'categorical' if var_to_choices.get(variable, '') else 'continuous'
        }
    
    # Save as JSON for easy loading in statistical software
    labels_file = os.path.join(output_dir, 'variable_labels.json')
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
    
    # Also create a CSV version for easier reading
    labels_df = pd.DataFrame([
        {
            'variable': var,
            'label': info['label'],
            'type': info['type'],
            'choices': info['choices']
        }
        for var, info in labels_data.items()
    ])
    
    labels_csv = os.path.join(output_dir, 'variable_labels.csv')
    labels_df.to_csv(labels_csv, index=False, encoding='utf-8')
    
    print(f"Variable labels saved to:")
    print(f"  JSON: {labels_file}")
    print(f"  CSV: {labels_csv}")
    
    return labels_data

def main():
    print("MAPP Data Separator - De-identified with Variable Labels")
    print("=" * 65)
    
    # Paths
    data_path = "C:/Users/ebweber/Claude/MAPP/data/MAPP_DATA_2025-08-20_0816.csv"
    dict_path = "C:/Users/ebweber/Claude/MAPP/metadata/MAPP_DataDictionary_2025-08-14.csv"
    
    # Load data and metadata
    print(f"Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Load metadata
    try:
        var_to_form, var_to_label, var_to_choices, hipaa_identifiers = load_and_process_metadata(dict_path)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return False
    
    # Create form-event mapping
    form_event_mapping = create_form_to_event_mapping()
    
    # Create output directory
    date_label = datetime.now().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(data_path)
    output_dir = os.path.join(base_dir, f"deidentified_labeled_data_{date_label}")
    
    # Create subdirectories
    events_dir = os.path.join(output_dir, 'events')
    ema_dir = os.path.join(output_dir, 'ema_combined')
    saliva_dir = os.path.join(output_dir, 'saliva_data')
    metadata_dir = os.path.join(output_dir, 'metadata')
    
    for dir_path in [output_dir, events_dir, ema_dir, saliva_dir, metadata_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created directories in: {output_dir}")
    
    # Create variable labels files
    print("\n=== CREATING VARIABLE LABELS ===")
    labels_data = create_variable_labels_file(var_to_label, var_to_choices, metadata_dir)
    
    # Get enrolled participants
    enrolled_participants = df[df['icf_staff_date'].notna()]['subject_id'].unique()
    print(f"\nFound {len(enrolled_participants)} enrolled participants")
    
    # 1. SEPARATE BY EVENTS 
    print(f"\n=== SEPARATING EVENTS (DE-IDENTIFIED & LABELED) ===")
    events = df['redcap_event_name'].dropna().unique()
    
    event_summary = {}
    
    for event in events:
        print(f"\nProcessing event: {event}")
        
        # Get variables for this event
        event_variables = get_variables_for_event(event, var_to_form, form_event_mapping)
        
        # Filter data for this event
        event_data = df[df['redcap_event_name'] == event].copy()
        
        # Apply enrollment filtering for baseline and enrollment events
        if event in ['enrollment_arm_1', 'baseline_preop_arm_1']:
            before_filter = len(event_data)
            event_data = event_data[event_data['subject_id'].isin(enrolled_participants)].copy()
            after_filter = len(event_data)
            print(f"  Enrollment filter: {before_filter} -> {after_filter} records")
        
        # Keep only appropriate variables
        available_vars = [var for var in event_variables if var in df.columns]
        event_data = event_data[available_vars].copy()
        
        # Remove HIPAA identifiers (except subject_id)
        event_data = remove_hipaa_identifiers(event_data, hipaa_identifiers)
        
        # Remove completely empty columns
        event_data = event_data.dropna(axis=1, how='all')
        
        # Clean for ASCII
        event_data = clean_for_ascii(event_data)
        
        # Save
        clean_name = event.replace('_', '-').replace(' ', '-')
        filename = f"{clean_name}_deidentified_{date_label}.csv"
        filepath = os.path.join(events_dir, filename)
        
        event_data.to_csv(filepath, index=False)
        print(f"  Saved: {event_data.shape[0]} records, {event_data.shape[1]} variables -> {filename}")
        
        event_summary[event] = {
            'records': event_data.shape[0],
            'variables': event_data.shape[1],
            'hipaa_removed': len([col for col in available_vars if col in hipaa_identifiers]),
            'enrollment_filtered': event in ['enrollment_arm_1', 'baseline_preop_arm_1']
        }
    
    # 2. COMBINE EMA DATA
    print(f"\n=== COMBINING EMA DATA (DE-IDENTIFIED) ===")
    
    ema_variables = get_variables_for_event('baseline_ema_morni_arm_1', var_to_form, form_event_mapping)
    ema_vars_in_data = [var for var in ema_variables if var in df.columns]
    
    ema_data = df[df['redcap_repeat_instrument'] == 'ema_questionnaire'].copy()
    
    if not ema_data.empty:
        # Filter to EMA variables and remove HIPAA identifiers
        ema_data = ema_data[ema_vars_in_data].copy()
        ema_data = remove_hipaa_identifiers(ema_data, hipaa_identifiers)
        ema_data = ema_data.dropna(axis=1, how='all')
        
        # Baseline EMA
        baseline_ema_events = ['baseline_ema_morni_arm_1', 'baseline_ema_after_arm_1', 'baseline_ema_eveni_arm_1']
        baseline_ema = ema_data[ema_data['redcap_event_name'].isin(baseline_ema_events)].copy()
        
        if not baseline_ema.empty:
            baseline_ema = clean_for_ascii(baseline_ema)
            baseline_path = os.path.join(ema_dir, f"ema_baseline_deidentified_{date_label}.csv")
            baseline_ema.to_csv(baseline_path, index=False)
            print(f"  Baseline EMA: {baseline_ema.shape[0]} records, {baseline_ema.shape[1]} variables")
        
        # 4-week EMA
        week4_ema_events = ['4_wk_ema_morning_arm_1', '4_wk__ema_afternoo_arm_1', '4_wk__ema_evening_arm_1']
        week4_ema = ema_data[ema_data['redcap_event_name'].isin(week4_ema_events)].copy()
        
        if not week4_ema.empty:
            week4_ema = clean_for_ascii(week4_ema)
            week4_path = os.path.join(ema_dir, f"ema_4week_deidentified_{date_label}.csv")
            week4_ema.to_csv(week4_path, index=False)
            print(f"  4-Week EMA: {week4_ema.shape[0]} records, {week4_ema.shape[1]} variables")
    
    # 3. EXTRACT SALIVA DATA (BY EVENT)
    print(f"\n=== EXTRACTING SALIVA DATA BY EVENT (DE-IDENTIFIED) ===")
    saliva_cols = [col for col in df.columns if 'sal_' in col.lower()]
    
    if saliva_cols:
        base_cols = ['subject_id', 'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance']
        saliva_data = df[base_cols + saliva_cols].copy()
        saliva_data = saliva_data.dropna(subset=saliva_cols, how='all')
        
        # Remove HIPAA identifiers from saliva data
        saliva_data = remove_hipaa_identifiers(saliva_data, hipaa_identifiers)
        saliva_data = clean_for_ascii(saliva_data)
        
        # Save combined file (for compatibility)
        complete_path = os.path.join(saliva_dir, f"saliva_complete_deidentified_{date_label}.csv")
        saliva_data.to_csv(complete_path, index=False)
        print(f"  Saliva complete: {saliva_data.shape[0]} records, {saliva_data.shape[1]} variables")
        
        # SEPARATE BY EVENT (like other data)
        saliva_events = saliva_data['redcap_event_name'].unique()
        for event in saliva_events:
            event_saliva = saliva_data[saliva_data['redcap_event_name'] == event].copy()
            
            if not event_saliva.empty:
                # Clean event name for filename
                clean_event_name = event.replace('_arm_1', '').replace('_', '-')
                event_filename = f"saliva-{clean_event_name}_deidentified_{date_label}.csv"
                event_path = os.path.join(events_dir, event_filename)
                
                event_saliva.to_csv(event_path, index=False)
                print(f"  {event}: {event_saliva.shape[0]} records -> {event_filename}")
    
    # 4. CREATE COMPREHENSIVE SUMMARY
    summary_file = os.path.join(metadata_dir, f"deidentification_summary_{date_label}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MAPP Data Separation - De-identified with Variable Labels\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Date: {date_label}\n")
        f.write(f"Original file: {data_path}\n")
        f.write(f"Original shape: {df.shape}\n")
        f.write(f"Enrolled participants: {len(enrolled_participants)}\n")
        f.write(f"HIPAA identifiers removed: {len(hipaa_identifiers)}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("HIPAA Identifiers Removed (except subject_id):\n")
        f.write("-" * 50 + "\n")
        for identifier in sorted(hipaa_identifiers):
            label = var_to_label.get(identifier, 'No label')
            f.write(f"  {identifier}: {label}\n")
        
        f.write(f"\nEvent Processing Summary:\n")
        f.write("-" * 25 + "\n")
        for event, stats in event_summary.items():
            f.write(f"\n{event}:\n")
            f.write(f"  Records: {stats['records']}\n")
            f.write(f"  Variables: {stats['variables']}\n")
            f.write(f"  HIPAA identifiers removed: {stats['hipaa_removed']}\n")
            if stats['enrollment_filtered']:
                f.write(f"  Note: Filtered to enrolled participants only\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")
    
    print("\n" + "=" * 65)
    print("DE-IDENTIFIED DATA SEPARATION WITH LABELS COMPLETE")
    print("=" * 65)
    print(f"Output directory: {output_dir}")
    print(f"Variable labels available in: {metadata_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("De-identified data separation failed!")