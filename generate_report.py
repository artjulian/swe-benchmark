import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_result_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract instance ID from filename
    instance_id = data['instance_id']
    
    # Check if results are empty
    if not data.get('results'):
        return {
            'instance_id': instance_id,
            'total_tests': 0,
            'passed_tests': 0,
            'fail_to_pass': 0,
            'pass_to_pass': 0,
            'has_warning': False
        }
    
    # Count total tests and passed tests
    total_tests = len(data['results'])
    passed_tests = sum(1 for result in data['results'] if result['exit_code'] == 0)
    
    # Count test types
    fail_to_pass = sum(1 for result in data['results'] if result['test_type'] == 'FAIL_TO_PASS' and result['exit_code'] == 0)
    pass_to_pass = sum(1 for result in data['results'] if result['test_type'] == 'PASS_TO_PASS' and result['exit_code'] == 0)
    
    # Check for any non-standard exit codes
    has_warning = any(result['exit_code'] not in [0, 1] for result in data['results'])
    
    return {
        'instance_id': instance_id,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'fail_to_pass': fail_to_pass,
        'pass_to_pass': pass_to_pass,
        'has_warning': has_warning
    }

def generate_html_report():
    # Get all result files
    results_dir = Path('results')
    result_files = list(results_dir.glob('result_*.json'))
    
    # Group results by repository
    repo_results = defaultdict(list)
    for file_path in result_files:
        repo_name = file_path.stem.split('-')[0].replace('result_', '')
        result_data = parse_result_file(file_path)
        repo_results[repo_name].append(result_data)
    
    # Calculate summary statistics
    total_instances = 0
    total_passed_instances = 0
    total_warnings = 0
    total_fail_to_pass = 0
    total_pass_to_pass = 0
    
    for results in repo_results.values():
        for result in results:
            total_instances += 1
            if result['passed_tests'] == result['total_tests']:
                total_passed_instances += 1
            if result['has_warning']:
                total_warnings += 1
            total_fail_to_pass += result['fail_to_pass']
            total_pass_to_pass += result['pass_to_pass']
    
    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SWE Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f5f5f5; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .pass { color: green; }
            .fail { color: red; }
            .warning { color: orange; }
            .summary { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .summary h2 { margin-top: 0; }
            .summary p { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>SWE Benchmark Results</h1>
    """
    
    # Add summary section at the top
    instance_pass_rate = (total_passed_instances / total_instances * 100) if total_instances > 0 else 0
    total_failed_instances = total_instances - total_passed_instances
    
    html += f"""
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p>Total Instances: {total_instances}</p>
            <p>Passed Instances: {total_passed_instances} ({instance_pass_rate:.1f}%)</p>
            <p>Failed Instances: {total_failed_instances} ({100 - instance_pass_rate:.1f}%)</p>
            <p>Total Warnings: {total_warnings}</p>
            <p>Total FAIL_TO_PASS passed: {total_fail_to_pass}</p>
            <p>Total PASS_TO_PASS passed: {total_pass_to_pass}</p>
        </div>
    """
    
    # Add tables for each repository
    for repo_name, results in repo_results.items():
        html += f"""
        <h2>Repository: {repo_name}</h2>
        <table>
            <tr>
                <th>Instance ID</th>
                <th>Total Tests</th>
                <th>Passed Tests</th>
                <th>FAIL_TO_PASS passed</th>
                <th>PASS_TO_PASS passed</th>
                <th>Status</th>
            </tr>
        """
        
        for result in results:
            # Mark as FAIL if there are no tests or if not all tests passed
            status_class = 'pass' if result['total_tests'] > 0 and result['passed_tests'] == result['total_tests'] else 'fail'
            status_text = 'PASS' if result['total_tests'] > 0 and result['passed_tests'] == result['total_tests'] else 'FAIL'
            
            if result['has_warning']:
                status_class = 'warning'
                status_text += ' (WARNING)'
            
            html += f"""
            <tr>
                <td>{result['instance_id']}</td>
                <td>{result['total_tests']}</td>
                <td>{result['passed_tests']}</td>
                <td>{result['fail_to_pass']}</td>
                <td>{result['pass_to_pass']}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
            """
        
        html += "</table>"
    
    html += """
    </body>
    </html>
    """
    
    # Create reports directory if it doesn't exist
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'benchmark_report_{timestamp}.html'
    
    # Write the HTML file
    report_path = reports_dir / report_filename
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {report_path}")

if __name__ == '__main__':
    generate_html_report() 