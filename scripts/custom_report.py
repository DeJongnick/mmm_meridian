import os
import pickle
import yaml
import re
import json
from pathlib import Path
from datetime import datetime

# Get POC directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
POC_DIR = SCRIPT_DIR.parent.absolute()


def list_saved_models():
    """List all saved models in outputs/models/"""
    models_dir = os.path.join(POC_DIR, "outputs", "models")
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for model_folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_folder)
        if os.path.isdir(model_path):
            metadata_path = os.path.join(model_path, "metadata.yaml")
            model_pkl_path = os.path.join(model_path, "model.pkl")
            
            # Check that model.pkl exists
            if not os.path.exists(model_pkl_path):
                continue
                
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    # Use FullLoader to support Python types like tuples
                    metadata = yaml.load(f, Loader=yaml.FullLoader)
                models.append({
                    "folder": model_folder,
                    "path": model_path,
                    "metadata": metadata,
                })
            else:
                # If no metadata, use the folder name as creation date
                models.append({
                    "folder": model_folder,
                    "path": model_path,
                    "metadata": {"created_at": model_folder},
                })
    
    # Sort by creation date (most recent first)
    return sorted(models, key=lambda x: x["folder"], reverse=True)


def load_model_from_pkl(model_path):
    """Load the model from the model.pkl file"""
    model_pkl_path = os.path.join(model_path, "model.pkl")
    
    if not os.path.exists(model_pkl_path):
        raise FileNotFoundError(f"model.pkl file does not exist in {model_path}")
    
    print(f"📂 Loading model from: {model_pkl_path}")
    with open(model_pkl_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    
    return model


def interactive_select_model():
    """Interactive menu to select a model"""
    models = list_saved_models()
    
    if not models:
        print("❌ No saved model found in outputs/models/")
        exit(1)
    
    print("\n" + "="*80)
    print("📚 MODEL SELECTION")
    print("="*80)
    print("\nAvailable models:\n")
    
    for i, model_info in enumerate(models, 1):
        meta = model_info["metadata"]
        folder = model_info["folder"]
        created_at = meta.get('created_at', folder)
        
        print(f"  {i}. 📅 {folder}")
        if 'data_shape' in meta:
            shape = meta.get('data_shape', 'N/A')
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                print(f"     📊 Data: {shape[0]} rows × {shape[1]} columns")
            date_range = meta.get('date_range', {})
            if date_range:
                start = date_range.get('start', 'N/A')
                end = date_range.get('end', 'N/A')
                if start != 'N/A' and end != 'N/A':
                    print(f"     📅 Period: {start[:10]} → {end[:10]}")
        print()
    
    while True:
        try:
            choice = input(f"Choose a model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"\n✓ Selected model: {selected['folder']}\n")
                return selected
            else:
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
            exit(1)


def extract_r2_from_html(report_html_path):
    """Extract R² score from report_data.html file"""
    try:
        if not os.path.exists(report_html_path):
            return None
        
        with open(report_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Search for R² in the metrics table
        # Pattern: after <th>R-squared</th>, get value in next <td>
        pattern = r'<th>R-squared</th>.*?<td[^>]*>([0-9.]+)</td>'
        match = re.search(pattern, html_content, re.DOTALL)
        
        if match:
            r2_str = match.group(1)
            try:
                r2 = float(r2_str)
                return r2
            except ValueError:
                pass
        
        # Alternative pattern
        pattern2 = r'R-squared.*?<td[^>]*>([0-9.]+)</td>'
        match2 = re.search(pattern2, html_content, re.IGNORECASE | re.DOTALL)
        
        if match2:
            r2_str = match2.group(1)
            try:
                r2 = float(r2_str)
                return r2
            except ValueError:
                pass
        
        return None
    except Exception as e:
        print(f"⚠️  Error extracting R² from HTML: {e}")
        return None


def extract_roi_by_channel(report_html_path):
    """Extract ROI by channel from report_data.html file"""
    try:
        if not os.path.exists(report_html_path):
            return {}
        
        with open(report_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        roi_data = {}
        all_channels = set()
        
        # Patterns for finding channel and roi (escaped or not)
        patterns = [
            r'\\"channel\\":\s*\\"([^\\"]+)\\"[^}]*\\"roi\\":\s*([0-9.]+)',
            r'"channel":\s*"([^"]+)"[^}]*"roi":\s*([0-9.]+)',
            r'\\"roi\\":\s*([0-9.]+)[^}]*\\"channel\\":\s*\\"([^\\"]+)\\"',
            r'"roi":\s*([0-9.]+)[^}]*"channel":\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html_content, re.DOTALL)
            for match in matches:
                # Determine which group is channel and which is roi
                if len(match.groups()) == 2:
                    if re.match(r'[A-Za-z]', match.group(1)):
                        channel = match.group(1).strip()
                        roi_str = match.group(2).strip()
                    else:
                        roi_str = match.group(1).strip()
                        channel = match.group(2).strip()
                    
                    roi_str = roi_str.rstrip('.').rstrip(',')
                    roi_str = re.sub(r'[^0-9.]', '', roi_str)
                    
                    try:
                        roi_value = float(roi_str)
                        if channel.upper() != "BASELINE":
                            if channel not in roi_data:
                                roi_data[channel] = roi_value
                            all_channels.add(channel)
                    except ValueError:
                        if channel.upper() != "BASELINE":
                            all_channels.add(channel)
                        continue
        
        # Find all unique channels (even if no ROI)
        channel_patterns = [
            r'\\"channel\\":\s*\\"([^\\"]+)\\"',
            r'"channel":\s*"([^"]+)"',
        ]
        for pattern in channel_patterns:
            channel_matches = re.finditer(pattern, html_content)
            for match in channel_matches:
                channel = match.group(1).strip()
                if channel.upper() != "BASELINE":
                    all_channels.add(channel)
        
        # Ensure all channels in roi_data, set None if no ROI
        for channel in all_channels:
            if channel not in roi_data:
                roi_data[channel] = None
        
        return roi_data
    except Exception as e:
        print(f"⚠️  Error extracting ROI from HTML: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_roi_html(roi_by_channel):
    """Generates HTML for ROI visualization by channel"""
    if not roi_by_channel:
        return '<div class="placeholder">Results will be displayed here</div>'
    
    channels_with_roi = [(ch, roi) for ch, roi in roi_by_channel.items() if roi is not None]
    
    if not channels_with_roi:
        return '<div class="placeholder">Results will be displayed here</div>'
    
    channels_with_roi.sort(key=lambda x: x[1], reverse=True)
    
    roi_items = []
    for channel, roi in channels_with_roi:
        roi_items.append(f'''
                <div class="roi-item">
                  <div class="roi-item-content">
                    <div class="roi-channel-name">{channel}</div>
                    <div class="roi-description">
                      For <strong>$1 invested</strong> in {channel} → ROI = <strong>${roi:.2f}</strong>
                    </div>
                  </div>
                  <div class="roi-value-container">
                    <div class="roi-value">{roi:.2f}</div>
                    <div class="roi-label">ROI</div>
                  </div>
                </div>
                ''')
    return f'''
              <div class="roi-visualization">
                {''.join(roi_items)}
              </div>
              '''


def extract_contribution_channel_chart_html(report_html_path):
    """Extract the full HTML section of the Contribution Channel chart from report_data.html (including baseline)"""
    try:
        if not os.path.exists(report_html_path):
            return None
        
        with open(report_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        pos = html_content.find('channel-drivers-chart')
        if pos == -1:
            return None
        
        start = html_content.rfind('<chart>', 0, pos)
        if start == -1:
            start = html_content.rfind('<chart-embed', 0, pos)
        if start == -1:
            return None
        
        end = html_content.find('</script>', pos)
        if end == -1:
            return None
        end += len('</script>')
        
        chart_section = html_content[start:end]
        
        # Edit JSON to remove the title and update chart palette
        json_pattern = r'const spec = JSON\.parse\("(.*?)"\);'
        match = re.search(json_pattern, chart_section, re.DOTALL)
        
        if match:
            json_str_escaped = match.group(1)
            import codecs
            try:
                json_str = codecs.decode(json_str_escaped, 'unicode_escape')
            except:
                json_str = json_str_escaped.replace('\\\\"', '"').replace('\\\\n', '\n').replace('\\\\\\\\', '\\')
            
            spec = json.loads(json_str)
            
            # Color palette updates
            if 'layer' in spec:
                for layer in spec['layer']:
                    if 'encoding' in layer and 'color' in layer['encoding']:
                        color_encoding = layer['encoding']['color']
                        if 'condition' in color_encoding:
                            if 'test' in color_encoding['condition'] and 'BASELINE' in color_encoding['condition']['test']:
                                color_encoding['condition']['value'] = '#8b5cf6'  # Secondary (violet)
                            if 'value' in color_encoding:
                                color_encoding['value'] = '#6366f1'  # Primary (indigo)
                        elif 'scale' in color_encoding and 'range' in color_encoding['scale']:
                            domain = color_encoding['scale'].get('domain', [])
                            colors_map = {
                                'BASELINE': '#8b5cf6',
                                'FACEBOOK': '#6366f1',
                                'GOOGLE ADS': '#818cf8',
                                'TIKTOK': '#10b981',
                            }
                            new_range = []
                            for domain_val in domain:
                                found = False
                                for key, color in colors_map.items():
                                    if key in str(domain_val).upper():
                                        new_range.append(color)
                                        found = True
                                        break
                                if not found:
                                    new_range.append('#6366f1')
                            if new_range:
                                color_encoding['scale']['range'] = new_range
            
            if 'title' in spec:
                spec['title'] = None
            
            new_json_str = json.dumps(spec)
            new_json_str_escaped = new_json_str.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            chart_section = chart_section.replace(json_str_escaped, new_json_str_escaped)
        
        chart_section = re.sub(r'<chart-description>.*?</chart-description>', '', chart_section, flags=re.DOTALL)
        chart_section = chart_section.replace('id="channel-drivers-chart"', 'id="contribution-channel-chart"')
        chart_section = chart_section.replace('#channel-drivers-chart', '#contribution-channel-chart')
        chart_section = chart_section.replace('channel-drivers-chart', 'contribution-channel-chart')
        
        return chart_section
        
    except Exception as e:
        print(f"⚠️ Error extracting Contribution Channel chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_model_fit_chart_html(report_html_path):
    """Extract the full HTML section of the Model Fit chart from report_data.html and remove the baseline"""
    try:
        if not os.path.exists(report_html_path):
            return None
        
        with open(report_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        pos = html_content.find('expected-actual-outcome-chart')
        if pos == -1:
            return None
        
        start = html_content.rfind('<chart>', 0, pos)
        if start == -1:
            start = html_content.rfind('<chart-embed', 0, pos)
        if start == -1:
            return None
        
        end = html_content.find('</script>', pos)
        if end == -1:
            return None
        end += len('</script>')
        
        chart_section = html_content[start:end]
        
        json_pattern = r'const spec = JSON\.parse\("(.*?)"\);'
        match = re.search(json_pattern, chart_section, re.DOTALL)
        
        if match:
            json_str_escaped = match.group(1)
            import codecs
            try:
                json_str = codecs.decode(json_str_escaped, 'unicode_escape')
            except:
                json_str = json_str_escaped.replace('\\\\"', '"').replace('\\\\n', '\n').replace('\\\\\\\\', '\\')
            
            spec = json.loads(json_str)
            
            # Filter out baseline
            if 'datasets' in spec:
                for dataset_name, dataset_data in spec['datasets'].items():
                    spec['datasets'][dataset_name] = [
                        item for item in dataset_data 
                        if item.get('type') != 'baseline'
                    ]
            
            # Update color palettes and remove baseline from domain
            if 'layer' in spec:
                for layer in spec['layer']:
                    if 'encoding' in layer and 'color' in layer['encoding']:
                        color_encoding = layer['encoding']['color']
                        if 'scale' in color_encoding and 'domain' in color_encoding['scale']:
                            color_encoding['scale']['domain'] = [
                                d for d in color_encoding['scale']['domain'] 
                                if d != 'baseline'
                            ]
                            if 'range' in color_encoding['scale']:
                                domain = color_encoding['scale']['domain']
                                new_range = []
                                for domain_val in domain:
                                    if 'expected' in domain_val.lower():
                                        new_range.append('#6366f1')
                                    elif 'actual' in domain_val.lower():
                                        new_range.append('#10b981')
                                    else:
                                        new_range.append('#6366f1')
                                color_encoding['scale']['range'] = new_range[:len(domain)]
                        elif 'condition' in color_encoding:
                            if 'value' in color_encoding:
                                color_encoding['value'] = '#6366f1'
                            if 'condition' in color_encoding and 'value' in color_encoding['condition']:
                                test = color_encoding['condition'].get('test', '')
                                if 'expected' in test.lower():
                                    color_encoding['condition']['value'] = '#6366f1'
                                elif 'actual' in test.lower():
                                    color_encoding['condition']['value'] = '#10b981'
            
            if 'title' in spec:
                spec['title'] = None
            
            new_json_str = json.dumps(spec)
            new_json_str_escaped = new_json_str.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            chart_section = chart_section.replace(json_str_escaped, new_json_str_escaped)
        
        chart_section = re.sub(r'<chart-description>.*?</chart-description>', '', chart_section, flags=re.DOTALL)
        chart_section = chart_section.replace('id="expected-actual-outcome-chart"', 'id="model-fit-chart"')
        chart_section = chart_section.replace('#expected-actual-outcome-chart', '#model-fit-chart')
        chart_section = chart_section.replace('expected-actual-outcome-chart', 'model-fit-chart')
        
        return chart_section
        
    except Exception as e:
        print(f"⚠️ Error extracting Model Fit chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_marketing_insights(r2_score, roi_by_channel):
    """Generates actionable insights for the marketing team based on model data"""
    insights = []
    
    # R² score analysis
    if r2_score is not None:
        if r2_score >= 0.75:
            pass  # No insight for excellent models
        elif r2_score >= 0.5:
            improvement_potential = int((0.75 - r2_score) * 100)
            insights.append({
                "type": "warning",
                "title": "⚠️ Model needs improvement for better accuracy",
                "description": f"With an R² of {r2_score:.3f}, the model explains a significant portion of the variation but can be improved by approximately {improvement_potential} points.",
                "action": "Enrich your data (external variables, seasonality, events) to improve model accuracy."
            })
        else:
            improvement_needed = int((0.5 - r2_score) * 100)
            insights.append({
                "type": "danger",
                "title": "🔴 Model requires revision",
                "description": f"With an R² of {r2_score:.3f}, the model explains less than half of the variation. It requires approximately {improvement_needed} points of improvement to be reliable.",
                "action": "Review model variables and collect more data to improve prediction quality."
            })
    
    # ROI by channel analysis
    if roi_by_channel:
        sorted_channels = sorted(
            [(ch, roi) for ch, roi in roi_by_channel.items() if roi is not None and roi != 'N/A'],
            key=lambda x: x[1],
            reverse=True
        )
        
        if len(sorted_channels) > 0:
            best_channel, best_roi = sorted_channels[0]
            worst_channel, worst_roi = sorted_channels[-1]
            
            if best_roi > 1.5:
                increase_pct = min(30, int((best_roi - 1.0) * 20))
                if increase_pct < 10:
                    increase_pct = 10
                insights.append({
                    "type": "success",
                    "title": f"🚀 {best_channel}: High-performing channel to prioritize",
                    "description": f"With a ROI of {best_roi:.2f}, every dollar invested in {best_channel} generates ${best_roi:.2f} in revenue. This is your most profitable channel.",
                    "action": f"Gradually increase the budget allocated to {best_channel} by {increase_pct}% to maximize return on investment."
                })
            elif best_roi > 1.0:
                increase_pct = min(15, int((best_roi - 1.0) * 30))
                if increase_pct < 5:
                    increase_pct = 5
                insights.append({
                    "type": "info",
                    "title": f"📈 {best_channel}: Profitable channel",
                    "description": f"With a ROI of {best_roi:.2f}, {best_channel} generates a positive return on investment.",
                    "action": f"Maintain or slightly increase the {best_channel} budget by {increase_pct}% while testing new creative approaches."
                })
            else:
                insights.append({
                    "type": "warning",
                    "title": f"⚠️ All channels underperforming",
                    "description": f"Even the best channel ({best_channel}) has a ROI of {best_roi:.2f}, below 1.0.",
                    "action": "Review your creative strategies, targets, and messaging. Test new approaches before increasing budgets."
                })
            
            # Insight on the worst performing channel
            if len(sorted_channels) > 1:
                performance_gap = best_roi - worst_roi
                performance_ratio = worst_roi / best_roi if best_roi > 0 else 0
                
                if performance_ratio < 0.7:
                    reduction_pct = min(40, int((1 - performance_ratio) * 50))
                    if reduction_pct < 15:
                        reduction_pct = 15
                    
                    reallocation_pct = min(35, int(performance_gap * 25))
                    if reallocation_pct < 20:
                        reallocation_pct = 20
                    
                    insights.append({
                        "type": "warning",
                        "title": f"🔍 {worst_channel}: Channel to optimize",
                        "description": f"With a ROI of {worst_roi:.2f}, {worst_channel} is {performance_gap:.2f} points below {best_channel} (ROI: {best_roi:.2f}), {int((1 - performance_ratio) * 100)}% less performant.",
                        "action": f"Analyze {worst_channel} performance: targeted audiences, creatives, placements. Reduce budget by {reduction_pct}% and reallocate {reallocation_pct}% to {best_channel}."
                    })
            
            # Diversification insight
            if len(sorted_channels) >= 3:
                roi_values = [roi for _, roi in sorted_channels]
                roi_variance = max(roi_values) - min(roi_values)
                roi_range_pct = int((roi_variance / max(roi_values)) * 100) if max(roi_values) > 0 else 0
                
                if roi_variance > 0.3:
                    reallocation_pct = min(30, int(roi_range_pct / 3))
                    if reallocation_pct < 10:
                        reallocation_pct = 10
                    
                    insights.append({
                        "type": "info",
                        "title": "💼 Channel diversification",
                        "description": f"Your channels show varied ROI (from {worst_roi:.2f} to {best_roi:.2f}), with a gap of {roi_variance:.2f} points ({roi_range_pct}% variation), indicating optimization opportunities.",
                        "action": f"Reallocate {reallocation_pct}% of budget from underperforming channels to the most profitable ones, while maintaining minimal presence to test new opportunities."
                    })
    
    # General optimization insight
    if roi_by_channel and len([r for r in roi_by_channel.values() if r is not None and r != 'N/A']) > 0:
        valid_rois = [r for r in roi_by_channel.values() if r is not None and r != 'N/A']
        avg_roi = sum(valid_rois) / len(valid_rois)
        above_one_count = len([r for r in valid_rois if r > 1.0])
        total_count = len(valid_rois)
        success_rate = int((above_one_count / total_count) * 100) if total_count > 0 else 0
        
        if avg_roi > 1.2:
            insights.append({
                "type": "success",
                "title": "💰 Positive overall performance",
                "description": f"Your media mix generates on average ${avg_roi:.2f} in revenue for every dollar invested. {success_rate}% of your channels ({above_one_count}/{total_count}) are profitable.",
                "action": "Maintain this performance by continuing to optimize budgets toward the most performing channels and regularly testing new approaches."
            })
        elif avg_roi > 1.0:
            insights.append({
                "type": "info",
                "title": "📊 Moderate overall performance",
                "description": f"Your media mix generates on average ${avg_roi:.2f} in revenue for every dollar invested. {success_rate}% of your channels ({above_one_count}/{total_count}) are profitable.",
                "action": f"Optimize your budgets by reallocating to the most performing channels to improve the overall average to ${avg_roi * 1.1:.2f}."
            })
    
    return insights


def generate_insights_html(insights):
    """Generates HTML to display insights"""
    if not insights:
        return '''
              <div class="placeholder">
                No insights available at the moment. Insights will be automatically generated from model data.
              </div>
              '''
    
    html_items = []
    for insight in insights:
        html_items.append(f'''
                  <div class="insight-item {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-description">{insight['description']}</div>
                    <div class="insight-action">{insight['action']}</div>
                  </div>
                  ''')
    
    return f'''
              <div class="insights-container">
                {''.join(html_items)}
              </div>
              '''


def generate_html_template(model_info, model=None):
    """Generate a modern and attractive HTML page to present the results"""
    metadata = model_info["metadata"]
    folder = model_info["folder"]
    
    # Extract model information
    created_at = metadata.get('created_at', folder)
    date_range = metadata.get('date_range', {})
    data_shape = metadata.get('data_shape', [])
    model_config = metadata.get('model_config', {})
    
    # Extract the R² score from report_data.html
    report_html_path = os.path.join(model_info["path"], "report_data.html")
    r2_score = extract_r2_from_html(report_html_path)
    
    # Extract ROI by channel
    roi_by_channel = extract_roi_by_channel(report_html_path)
    
    # Extract Model Fit chart section
    model_fit_html = extract_model_fit_chart_html(report_html_path)
    
    # Extract Contribution Channel chart section
    contribution_channel_html = extract_contribution_channel_chart_html(report_html_path)
    
    # Generate actionable marketing insights
    marketing_insights = generate_marketing_insights(r2_score, roi_by_channel)
    
    # R² quality and summary for display
    r2_quality = None
    r2_quality_label = None
    r2_interpretation = None
    if r2_score is not None:
        if r2_score >= 0.75:
            r2_quality = "excellent"
            r2_quality_label = "Excellent"
            r2_interpretation = f"<strong>R² = {r2_score:.3f} (≥ 0.75)</strong>: The model explains at least 75% of the variation in your data. This is an <strong>excellent</strong> fit indicating that the model captures trends and relationships in your marketing data very well."
        elif r2_score >= 0.5:
            r2_quality = "good"
            r2_quality_label = "Good"
            r2_interpretation = f"<strong>R² = {r2_score:.3f} (0.5 - 0.75)</strong>: The model explains between 50% and 75% of the variation in your data. This is a <strong>good</strong> fit, but there is still room for improvement to better capture marketing relationships."
        else:
            r2_quality = "improve"
            r2_quality_label = "Needs Improvement"
            r2_interpretation = f"<strong>R² = {r2_score:.3f} (&lt; 0.5)</strong>: The model explains less than 50% of the variation in your data. The fit <strong>needs improvement</strong>. It would be beneficial to review model variables or enrich the data to better capture marketing relationships."
    
    # Format creation date
    try:
        if isinstance(created_at, str) and 'T' in created_at:
            dt = datetime.fromisoformat(created_at)
            formatted_date = dt.strftime("%d/%m/%Y at %H:%M")
        else:
            formatted_date = created_at
    except:
        formatted_date = created_at
    
    # Format period
    start_date = date_range.get('start', 'N/A')
    end_date = date_range.get('end', 'N/A')
    if start_date != 'N/A' and isinstance(start_date, str):
        start_date = start_date[:10] if len(start_date) > 10 else start_date
    if end_date != 'N/A' and isinstance(end_date, str):
        end_date = end_date[:10] if len(end_date) > 10 else end_date
    
    html_content = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>MMM Report - {folder}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
           <link rel="preconnect" href="https://fonts.googleapis.com" />
           <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
           <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
           <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
           <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
           <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
      :root {{
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --primary-light: #818cf8;
        --secondary: #8b5cf6;
        --secondary-dark: #7c3aed;
        --accent: #ec4899;
        --success: #10b981;
        --success-light: #34d399;
        --warning: #f59e0b;
        --warning-light: #fbbf24;
        --danger: #ef4444;
        --danger-light: #f87171;
        --info: #3b82f6;
        --info-light: #60a5fa;
        --bg: #f8fafc;
        --bg-card: #ffffff;
        --bg-gradient-start: #667eea;
        --bg-gradient-end: #764ba2;
        --text: #1e293b;
        --text-muted: #64748b;
        --text-light: #94a3b8;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-md: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 16px;
      }}

      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}

      body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%);
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: var(--text);
        line-height: 1.6;
        min-height: 100vh;
        padding: 2rem 1rem;
      }}

      @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
      }}

      .container {{
        max-width: 1400px;
        margin: 0 auto;
        animation: fadeIn 0.8s ease-out;
      }}

      @keyframes fadeIn {{
        from {{
          opacity: 0;
          transform: translateY(30px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}

      /* Header */
      .header {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg);
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.5);
      }}

      .header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
      }}

      .header-content h1 {{
        font-size: 2.75rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
      }}

      .subtitle {{
        font-size: 1.15rem;
        color: var(--text-muted);
        font-weight: 400;
        margin-bottom: 1.5rem;
      }}

      .header-meta {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
      }}

      .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
        color: white;
        padding: 0.625rem 1.25rem;
        border-radius: var(--radius);
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: var(--shadow-md);
        transition: transform 0.2s ease;
      }}

      .badge:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
      }}

      .badge-icon {{
        font-size: 1.1rem;
      }}

      /* Section */
      .section {{
        margin-bottom: 2.5rem;
      }}

      .section-title {{
        font-size: 1.875rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }}

      .section-title::before {{
        content: '';
        width: 5px;
        height: 28px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 3px;
        box-shadow: var(--shadow-sm);
      }}

      /* Grid */
      .grid {{
        display: grid;
        gap: 1.5rem;
      }}

      .grid.vertical {{
        grid-template-columns: 1fr;
      }}

      .grid-full {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 1.5rem;
      }}

      @media (min-width: 768px) {{
        .grid:not(.vertical) {{
          grid-template-columns: repeat(2, 1fr);
        }}
      }}

      /* Card */
      .card {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.5);
      }}

      .card:hover {{
        box-shadow: var(--shadow-xl);
        transform: translateY(-4px);
      }}

      .card-header {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-bottom: 2px solid var(--border-light);
      }}

      .card-icon {{
        font-size: 2rem;
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: var(--radius);
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease;
      }}

      .card:hover .card-icon {{
        transform: scale(1.1) rotate(5deg);
      }}

      .card-header h2 {{
        font-size: 1.375rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
      }}

      .card-content {{
        padding: 2rem;
      }}

      .chart-card {{
        min-height: 500px;
      }}

      /* Center visualizations */
      .chart-card .card-content {{
        display: flex;
        flex-direction: column;
        align-items: center;
      }}

      .chart-card .card-content > p {{
        text-align: center;
        max-width: 800px;
      }}

      .chart-card .card-content > div:not(.placeholder),
      .chart-card .card-content > chart,
      .chart-card .card-content > chart-embed,
      .chart-card .card-content #model-fit-chart,
      .chart-card .card-content #contribution-channel-chart {{
        width: 100%;
        max-width: 1000px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
      }}

      .chart-card .card-content #vega-vis,
      .chart-card .card-content .vega-embed {{
        display: flex;
        justify-content: center;
        margin: 0 auto;
      }}

      /* Stats */
      .stats-grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 1rem;
      }}

      .stat-card {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 2px solid rgba(99, 102, 241, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
      }}

      .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        transform: scaleX(0);
        transition: transform 0.3s ease;
      }}

      .stat-card:hover {{
        border-color: var(--primary);
        transform: scale(1.03);
        box-shadow: var(--shadow-md);
      }}

      .stat-card:hover::before {{
        transform: scaleX(1);
      }}

      .stat-value {{
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
      }}

      .stat-label {{
        font-size: 0.875rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
      }}

      .quality-badge {{
        display: inline-block;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        border-radius: var(--radius);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: var(--shadow-sm);
      }}

      .quality-badge.excellent {{
        background: linear-gradient(135deg, var(--success-light) 0%, var(--success) 100%);
        color: white;
      }}

      .quality-badge.good {{
        background: linear-gradient(135deg, var(--info-light) 0%, var(--info) 100%);
        color: white;
      }}

      .quality-badge.improve {{
        background: linear-gradient(135deg, var(--warning-light) 0%, var(--warning) 100%);
        color: white;
      }}

      /* Info Button */
      .info-button {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.625rem 1.25rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 2px solid var(--primary);
        border-radius: var(--radius);
        color: var(--primary);
        font-size: 0.875rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
      }}

      .info-button:hover {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
      }}

      .info-button svg {{
        width: 18px;
        height: 18px;
      }}

      /* R2 Dropdown */
      .r2-dropdown {{
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.5s ease, margin 0.5s ease;
        margin-top: 0;
      }}

      .r2-dropdown.open {{
        max-height: 600px;
        margin-top: 1.5rem;
      }}

      .r2-dropdown-content {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 2px solid rgba(99, 102, 241, 0.2);
        box-shadow: var(--shadow-sm);
      }}

      .r2-dropdown-content h3 {{
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text);
      }}

      .r2-dropdown-content p {{
        margin-bottom: 1rem;
        color: var(--text);
        line-height: 1.8;
      }}

      /* ROI Visualization */
      .roi-visualization {{
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }}

      .roi-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 1.75rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%);
        border: 2px solid rgba(16, 185, 129, 0.2);
        border-radius: var(--radius-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }}

      .roi-item::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(135deg, var(--success) 0%, var(--success-light) 100%);
        transform: scaleY(0);
        transition: transform 0.3s ease;
      }}

      .roi-item:hover {{
        border-color: var(--success);
        box-shadow: var(--shadow-md);
        transform: translateX(8px);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
      }}

      .roi-item:hover::before {{
        transform: scaleY(1);
      }}

      .roi-item-content {{
        flex: 1;
      }}

      .roi-channel-name {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.5rem;
      }}

      .roi-description {{
        font-size: 0.9375rem;
        color: var(--text-muted);
        line-height: 1.6;
      }}

      .roi-value-container {{
        text-align: center;
        padding-left: 2rem;
        border-left: 2px solid rgba(16, 185, 129, 0.3);
      }}

      .roi-value {{
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--success) 0%, var(--success-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
      }}

      .roi-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
      }}

      /* Insights */
      .insights-container {{
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
      }}

      .insight-item {{
        padding: 1.75rem;
        border-radius: var(--radius-lg);
        border-left: 5px solid;
        background: var(--bg);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }}

      .insight-item::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        opacity: 0;
        transition: opacity 0.3s ease;
      }}

      .insight-item:hover {{
        transform: translateX(8px);
        box-shadow: var(--shadow-md);
      }}

      .insight-item.success {{
        border-left-color: var(--success);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%);
      }}

      .insight-item.success::before {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%);
      }}

      .insight-item.info {{
        border-left-color: var(--info);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.03) 100%);
      }}

      .insight-item.info::before {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%);
      }}

      .insight-item.warning {{
        border-left-color: var(--warning);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%);
      }}

      .insight-item.warning::before {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, transparent 100%);
      }}

      .insight-item.danger {{
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.03) 100%);
      }}

      .insight-item.danger::before {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, transparent 100%);
      }}

      .insight-item:hover::before {{
        opacity: 1;
      }}

      .insight-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.75rem;
      }}

      .insight-description {{
        font-size: 0.9375rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
        line-height: 1.8;
      }}

      .insight-action {{
        font-size: 0.875rem;
        color: var(--text);
        font-weight: 600;
        padding: 0.875rem 1.25rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
      }}

      /* Placeholder */
      .placeholder {{
        padding: 4rem 2rem;
        text-align: center;
        color: var(--text-muted);
        font-style: italic;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: var(--radius-lg);
      }}

      /* Responsive */
      @media (max-width: 768px) {{
        body {{
          padding: 1rem 0.5rem;
        }}

        .header {{
          padding: 2rem 1.5rem;
        }}

        .header-content h1 {{
          font-size: 2rem;
        }}

        .card-content {{
          padding: 1.5rem;
        }}

        .roi-item {{
          flex-direction: column;
          align-items: flex-start;
        }}

        .roi-value-container {{
          border-left: none;
          border-top: 2px solid rgba(16, 185, 129, 0.3);
          padding-left: 0;
          padding-top: 1rem;
          margin-top: 1rem;
          width: 100%;
        }}

        .stat-value {{
          font-size: 2.5rem;
        }}

        .section-title {{
          font-size: 1.5rem;
        }}
      }}

      /* Chart container styling */
      .card-content p {{
        margin-bottom: 1.5rem;
        color: var(--text-muted);
        line-height: 1.8;
      }}

      /* Smooth scrolling */
      html {{
        scroll-behavior: smooth;
      }}

      /* Additional animations */
      @keyframes pulse {{
        0%, 100% {{
          opacity: 1;
        }}
        50% {{
          opacity: 0.7;
        }}
      }}

      .badge {{
        animation: pulse 3s ease-in-out infinite;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <div class="header-content">
          <h1>Media Mix Modeling Report</h1>
          <p class="subtitle">Detailed analysis of model performance</p>
                 <div class="header-meta">
                   <div class="badge">
                     <span class="badge-icon">🏷️</span>
                     <span>Model: {folder}</span>
                   </div>
                   {f'<div class="badge"><span class="badge-icon">📅</span><span>Period: {start_date} → {end_date}</span></div>' if start_date != 'N/A' and end_date != 'N/A' else ''}
                 </div>
        </div>
      </div>
      
      <div class="section">
        <h2 class="section-title">Overview</h2>
        <div class="grid">
          <div class="card">
            <div class="card-header">
              <div class="card-icon">⚡</div>
              <h2>Model Performance</h2>
            </div>
            <div class="card-content">
              {f'''
              <div class="stats-grid" style="margin-top: 24px;">
                <div class="stat-card">
                  <div class="stat-value">{r2_score:.3f}</div>
                  <div class="stat-label">R² Score</div>
                  {f'<div class="quality-badge {r2_quality}">✓ {r2_quality_label}</div>' if r2_quality else ''}
                </div>
              </div>
              
              <button class="info-button" onclick="toggleR2Info()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span>Info</span>
              </button>
              
              <div class="r2-dropdown" id="r2-dropdown">
                <div class="r2-dropdown-content">
                  <h3>📚 What is the R² Score?</h3>
                  <p>
                    The <strong>R² (coefficient of determination)</strong> measures the quality of the model's fit to observed data. 
                    It indicates the <strong>percentage of variation</strong> in your data that is explained by the model.
                  </p>
                  <p>
                    <strong>Your model interpretation:</strong><br>
                    {r2_interpretation if r2_interpretation else ''}
                  </p>
                  <p style="font-size: 0.85rem; font-style: italic; margin-top: 12px;">
                    The closer R² is to 1, the more accurately the model can predict observed results.
                  </p>
                </div>
              </div>
              ''' if r2_score is not None else '''
              <div class="placeholder">
                Results will be displayed here
              </div>
              '''}
            </div>
          </div>
          
          <div class="card">
            <div class="card-header">
              <div class="card-icon">💹</div>
              <h2>ROI by Media Channel</h2>
            </div>
            <div class="card-content">
              <p style="font-size: 0.95rem; color: var(--text-muted); margin-bottom: 24px;">
                Return on investment for each dollar spent per channel
              </p>
              {generate_roi_html(roi_by_channel)}
            </div>
          </div>
        </div>
      </div>
      
      <div class="section">
        <h2 class="section-title">Visualizations</h2>
        <div class="grid vertical">
          <div class="card chart-card">
            <div class="card-header">
              <div class="card-icon">📈</div>
              <h2>Model Fit</h2>
            </div>
            <div class="card-content">
              <p>Comparison between expected revenues from the model and observed actual revenues, allowing to evaluate prediction accuracy.</p>
              {model_fit_html if model_fit_html else '''
              <div class="placeholder">
                Model fit charts will be displayed here
              </div>
              '''}
            </div>
          </div>
          
          <div class="card chart-card">
            <div class="card-header">
              <div class="card-icon">📊</div>
              <h2>Contribution Channel</h2>
            </div>
            <div class="card-content">
              <p>Breakdown of each media channel's contribution (baseline and marketing channels) to overall performance, showing the relative impact of each source.</p>
              {contribution_channel_html if contribution_channel_html else '''
              <div class="placeholder">
                Contribution charts will be displayed here
              </div>
              '''}
            </div>
          </div>
        </div>
      </div>
      
      <div class="section">
        <h2 class="section-title">Insights</h2>
        <div class="grid-full">
          <div class="card">
            <div class="card-header">
              <div class="card-icon">💡</div>
              <h2>Actionable Recommendations</h2>
            </div>
            <div class="card-content">
              <p style="font-size: 0.95rem; color: var(--text-muted); margin-bottom: 24px;">
                Actionable recommendations based on your MMM model analysis to optimize your marketing decisions.
              </p>
              {generate_insights_html(marketing_insights)}
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <script>
      function toggleR2Info() {{
        const dropdown = document.getElementById('r2-dropdown');
        const button = event.currentTarget;
        
        if (dropdown.classList.contains('open')) {{
          dropdown.classList.remove('open');
          button.querySelector('span').textContent = 'Info';
        }} else {{
          dropdown.classList.add('open');
          button.querySelector('span').textContent = 'Hide';
        }}
      }}
    </script>
  </body>
</html>"""
    return html_content


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🎨 CUSTOM REPORT GENERATION")
    print("="*80)
    
    # Select a model
    model_info = interactive_select_model()
    
    # Load model from .pkl file
    print("\n" + "="*80)
    print("📂 LOADING MODEL")
    print("="*80)
    try:
        model = load_model_from_pkl(model_info["path"])
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)
    
    # Generate HTML
    print("\n" + "="*80)
    print("📝 GENERATING HTML PAGE")
    print("="*80)
    html_content = generate_html_template(model_info, model=model)
    
    # Save HTML file
    output_path = os.path.join(model_info["path"], "custom_report.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML report generated successfully!")
    print(f"   📁 Location: {output_path}")
    print(f"\n💡 You can open the file in your browser to view the report.")


if __name__ == "__main__":
    main()

