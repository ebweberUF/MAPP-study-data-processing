#!/usr/bin/env python3
"""
Create improved HTML report using the better-spaced visualizations
"""

import os
import base64
from datetime import datetime

def create_improved_html_report():
    """
    Create HTML report with improved visualizations
    """
    
    data_dir = "C:/Users/ebweber/Claude/MAPP/data/deidentified_labeled_data_2025-08-20"
    improved_reports_dir = os.path.join(data_dir, "improved_visual_reports")
    
    if not os.path.exists(improved_reports_dir):
        print(f"Improved reports directory not found: {improved_reports_dir}")
        return False
    
    def image_to_base64(image_path):
        """Convert image to base64 for embedding in HTML"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except:
            return ""
    
    date_generated = datetime.now().strftime("%B %d, %Y")
    
    # Define improved report sections
    sections = [
        {
            'title': 'Executive Summary Dashboard',
            'image': 'executive_summary_improved.png',
            'description': 'Comprehensive overview of study metrics, enrollment status, surgery distribution (knee vs hip procedures), retention rates, and project milestones with clear data labels and professional formatting.'
        },
        {
            'title': 'Baseline Patient Characteristics', 
            'image': 'baseline_demographics_improved.png',
            'description': 'Detailed demographic profile showing age distribution with data labels, sex distribution with counts, surgery site breakdown, and comprehensive study statistics in an easy-to-read format.'
        },
        {
            'title': 'PROMIS Patient-Reported Outcomes',
            'image': 'promis_outcomes_improved.png', 
            'description': 'Baseline distributions of validated PROMIS measures with clear histograms, data labels on bars, mean/median lines, and comprehensive statistics boxes for each measure.'
        },
        {
            'title': 'Pain Outcome Trajectories',
            'image': 'pain_trajectories_improved.png',
            'description': 'Longitudinal pain measure changes from pre- to post-surgery with larger markers, value labels at each timepoint, sample sizes, and trend lines showing clinical progression.'
        },
        {
            'title': 'Missing Data Analysis',
            'image': 'missing_data_analysis_improved.png',
            'description': 'Corrected analysis distinguishing true missing data from study design. Shows participant completion status, true within-assessment missingness rates, attrition flow, and explains REDCap study protocol vs. actual missing data patterns.'
        },
        {
            'title': 'Saliva Collection Analysis',
            'image': 'saliva_collection_analysis.png',
            'description': 'Comprehensive analysis of biomarker collection compliance. Shows participant-level collection status for 10 samples per timepoint (5 samples/day × 2 days), compliance rates, day-by-day collection patterns, and protocol adherence across enrollment and follow-up periods.'
        }
    ]
    
    # Create improved HTML content
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAPP Study - Improved Professional Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
            color: white;
            padding: 3rem 0;
            text-align: center;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            font-weight: 300;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.4rem;
            opacity: 0.95;
            font-weight: 300;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }}
        
        .improvement-notice {{
            background: linear-gradient(90deg, #28a745, #20c997);
            color: white;
            padding: 1.5rem;
            margin: 2rem auto;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .improvement-notice h2 {{
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
        }}
        
        .improvement-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .improvement-item {{
            background: rgba(255,255,255,0.2);
            padding: 1rem;
            border-radius: 6px;
            font-size: 0.95rem;
        }}
        
        .report-info {{
            background: white;
            margin: 2rem auto;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        
        .report-info h2 {{
            color: #1f77b4;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 8px;
            border-left: 5px solid #1f77b4;
            transition: transform 0.3s ease;
        }}
        
        .info-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        
        .info-card h3 {{
            color: #1f77b4;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }}
        
        .section {{
            background: white;
            margin: 3rem auto;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #1f77b4, #0d47a1);
            color: white;
            padding: 2rem 2.5rem;
        }}
        
        .section-header h2 {{
            font-size: 1.8rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }}
        
        .section-header .badge {{
            background: rgba(255,255,255,0.2);
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.9rem;
            display: inline-block;
        }}
        
        .section-content {{
            padding: 2.5rem;
        }}
        
        .section-description {{
            color: #555;
            margin-bottom: 2.5rem;
            font-size: 1.1rem;
            line-height: 1.8;
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #ff7f0e;
        }}
        
        .visualization {{
            text-align: center;
            margin: 3rem 0;
            background: #fafafa;
            padding: 2rem;
            border-radius: 12px;
            border: 2px solid #e9ecef;
        }}
        
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            transition: transform 0.3s ease;
        }}
        
        .visualization img:hover {{
            transform: scale(1.02);
        }}
        
        .stats-highlight {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 2rem;
            text-align: center;
            border-radius: 8px;
            border-top: 4px solid #ff7f0e;
            transition: transform 0.3s ease;
        }}
        
        .stat-box:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        
        .stat-number {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            display: block;
            margin-bottom: 0.5rem;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 1rem;
            font-weight: 500;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 3rem 0;
            margin-top: 4rem;
        }}
        
        .footer p {{
            opacity: 0.9;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2.2rem;
            }}
            
            .container {{
                padding: 0 1rem;
            }}
            
            .section-content {{
                padding: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>MAPP Study</h1>
            <div class="subtitle">Multi-modal Assessment of Pain Prediction<br>
            Professional Research Report</div>
        </div>
    </div>

    <div class="container">
        <div class="report-info">
            <h2>Study Overview</h2>
            <p>The MAPP (Multi-modal Assessment of Pain Prediction) study is a prospective observational research investigation designed to identify factors that predict pain and functional outcomes following knee and hip replacement surgery. This comprehensive report presents baseline characteristics, study design, and preliminary findings from enrolled participants.</p>
            
            <div class="info-grid">
                <div class="info-card">
                    <h3>Study Design</h3>
                    <p>Prospective observational cohort study with 9-month longitudinal follow-up</p>
                </div>
                <div class="info-card">
                    <h3>Setting</h3>
                    <p>University of Florida Health Orthopaedics and Sports Medicine Institute</p>
                </div>
                <div class="info-card">
                    <h3>Population</h3>
                    <p>Adults undergoing elective primary knee or hip arthroplasty</p>
                </div>
                <div class="info-card">
                    <h3>Report Generated</h3>
                    <p>{date_generated}</p>
                </div>
            </div>
        </div>
    '''
    
    # Add each improved section
    for i, section in enumerate(sections, 1):
        image_path = os.path.join(improved_reports_dir, section['image'])
        image_base64 = image_to_base64(image_path)
        
        html_content += f'''
        <div class="section" id="section{i}">
            <div class="section-header">
                <h2>{i}. {section['title']}</h2>
            </div>
            <div class="section-content">
                <div class="section-description">
                    {section['description']}
                </div>
        '''
        
        if image_base64:
            html_content += f'''
                <div class="visualization">
                    <img src="data:image/png;base64,{image_base64}" alt="{section['title']}" />
                </div>
            '''
        else:
            html_content += '''
                <div class="highlight">
                    <p><strong>Note:</strong> Enhanced visualization not available. Please ensure all improved image files are present.</p>
                </div>
            '''
        
        html_content += '''
            </div>
        </div>
        '''
    
    # Add study highlights section
    html_content += f'''
        <div class="section">
            <div class="section-header">
                <h2>Key Study Highlights</h2>
            </div>
            <div class="section-content">
                <div class="stats-highlight">
                    <div class="stat-box">
                        <span class="stat-number">21</span>
                        <div class="stat-label">Participants Enrolled</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">67.7</span>
                        <div class="stat-label">Mean Age (years)</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">67%</span>
                        <div class="stat-label">Female Participants</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">5</span>
                        <div class="stat-label">Assessment Timepoints</div>
                    </div>
                </div>
                
                <div class="section-description">
                    <strong>Study Significance:</strong> This research addresses a critical clinical need by developing predictive models for post-surgical pain outcomes. The multi-modal approach combining patient-reported outcomes, quantitative sensory testing, and biomarker analysis provides comprehensive insight into factors influencing recovery after joint replacement surgery.
                </div>
                
                <div class="section-description">
                    <strong>Clinical Impact:</strong> Results from this study will inform evidence-based approaches to pre-surgical risk stratification, personalized pain management strategies, patient counseling and expectation setting, and healthcare resource allocation and planning.
                </div>
            </div>
        </div>
    '''
    
    # Add footer
    html_content += f'''
    </div>

    <div class="footer">
        <div class="container">
            <p>MAPP Study - Multi-modal Assessment of Pain Prediction</p>
            <p>University of Florida Health &bull; Enhanced Report Generated: {date_generated}</p>
            <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;">
                Enhanced version with improved spacing, data labels, and professional formatting
            </p>
        </div>
    </div>

</body>
</html>
    '''
    
    # Save improved HTML report
    html_file = os.path.join(improved_reports_dir, 'MAPP_Study_Improved_Professional_Report.html')
    
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Improved HTML report created: {html_file}")
        return True
        
    except Exception as e:
        print(f"Error creating improved HTML report: {e}")
        return False

def main():
    """
    Main execution function
    """
    success = create_improved_html_report()
    
    if success:
        print("\\n" + "=" * 65)
        print("CLEAN HTML REPORT CREATED SUCCESSFULLY")
        print("=" * 65)
        print("\\nReport now focuses on study content with:")
        print("- Study overview and design")
        print("- Key visualizations")
        print("- Study highlights and significance")
        print("- Professional formatting")
        print("- Removed improvement sections as requested")
        
        return True
    else:
        print("Failed to create improved HTML report")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("Improved HTML report generation failed!")