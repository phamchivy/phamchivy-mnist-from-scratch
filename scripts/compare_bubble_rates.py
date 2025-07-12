import matplotlib.pyplot as plt
import re
import sys
import os

def parse_log_file(filename):
    """Parse a single log file and extract bubble rate data"""
    bubble_rates = []
    training_progress = []
    
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found")
        return [], []
    
    try:
        # Try UTF-16 first (for Windows logs)
        with open(filename, 'r', encoding='utf-16') as f:
            content = f.read()
    except UnicodeError:
        # Fall back to UTF-8
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    
    lines = content.split('\n')
    
    max_batch = 0
    # First pass: find maximum batch count
    for line in lines:
        if 'METRIC_LOG' in line and 'BUBBLE_RATE' in line:
            match = re.search(r'BUBBLE_RATE\|([0-9.]+)\|([0-9.]+)\|(\d+)', line)
            if match:
                batch_count = int(match.group(3))
                max_batch = max(max_batch, batch_count)
    
    # Second pass: extract bubble rates
    for line in lines:
        if 'METRIC_LOG' in line and 'BUBBLE_RATE' in line:
            match = re.search(r'BUBBLE_RATE\|([0-9.]+)\|([0-9.]+)\|(\d+)', line)
            if match:
                bubble_rate = float(match.group(1))
                batch_count = int(match.group(3))
                
                # Convert to training progress percentage
                if max_batch > 0:
                    progress = (batch_count / max_batch) * 100
                    training_progress.append(progress)
                    bubble_rates.append(bubble_rate)
    
    return training_progress, bubble_rates

def create_comparison_chart(log_files):
    """Create comparison chart for multiple log files"""
    plt.figure(figsize=(12, 8))
    
    # Configuration labels in order
    config_labels = ['Pipeline Depth=4', 'Pipeline Depth=8', 'Pipeline Depth=16']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, log_file in enumerate(log_files):
        if i >= len(config_labels):
            break
            
        training_progress, bubble_rates = parse_log_file(log_file)
        
        if training_progress and bubble_rates:
            label = config_labels[i]
            color = colors[i % len(colors)]
            
            plt.plot(training_progress, bubble_rates, 
                    label=label, color=color, linewidth=2, alpha=0.8)
            
            # Add average bubble rate to legend
            avg_rate = sum(bubble_rates) / len(bubble_rates)
            print(f"{label}: Average bubble rate = {avg_rate:.4f}")
        else:
            print(f"Warning: No data found in {log_file}")
    
    plt.xlabel('Training Progress (%)', fontsize=12)
    plt.ylabel('Bubble Rate', fontsize=12)
    plt.title('Pipeline Bubble Rate Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 100)
    plt.ylim(0, None)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('bubble_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nChart saved as 'bubble_rate_comparison.png'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_bubble_rates.py <log_file1> [log_file2] [log_file3]")
        print("Files should be provided in order: baseline, depth=4, depth=8")
        sys.exit(1)
    
    log_files = sys.argv[1:]
    
    if len(log_files) > 3:
        print("Warning: Only first 3 files will be processed")
        log_files = log_files[:3]
    
    print(f"Processing {len(log_files)} log files...")
    for i, filename in enumerate(log_files):
        config_names = ['Baseline', 'Pipeline Depth=4', 'Pipeline Depth=8']
        config_name = config_names[i] if i < len(config_names) else f'Config {i+1}'
        print(f"File {i+1}: {filename} -> {config_name}")
    
    create_comparison_chart(log_files)

if __name__ == "__main__":
    main() 