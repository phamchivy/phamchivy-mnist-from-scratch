#include "pipeline_nn.h"
#include "metrics_tracking.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// External global variables
extern int total_pipeline_bubbles;

// Implementation of update_pipeline_config function
void update_pipeline_config(PipelineConfig* config, PipelineStats* stats) {
    if (!config || !stats) return;
    
    // Calculate current bubble rate
    double current_bubble_rate = 0.0;
    if (stats->processed_batches > 0) {
        current_bubble_rate = (double)total_pipeline_bubbles / stats->processed_batches;
    }
    
    // Update bubble rate in config
    config->bubble_rate = current_bubble_rate * 100.0; // Convert to percentage
    
    // Calculate communication ratio
    if (stats->stage1_processing_time + stats->stage2_processing_time > 0) {
        double total_processing = stats->stage1_processing_time + stats->stage2_processing_time;
        config->communication_ratio = stats->communication_time / total_processing;
    }
    
    // Update last adjustment time
    config->last_adjustment_time = get_current_timestamp();
}

// Implementation of calculate_adaptive_batch_size function
int calculate_adaptive_batch_size(PipelineStats* stats, PipelineConfig* config) {
    if (!stats || !config) return MINI_BATCH_SIZE;
    
    int current_size = config->current_batch_size;
    int new_size = current_size;
    
    // Primary scaling based on bubble rate
    double bubble_rate = config->bubble_rate / 100.0; // Convert back to decimal
    
    // Bubble rate thresholds (in percentage for clarity)
    // 0-5%: Very low (increase +8)
    // 5-10%: Low (increase +4) 
    // 10-15%: Acceptable (no change)
    // 15-25%: Moderate (decrease -4)
    // 25-40%: High (decrease -8)
    // 40%+: Critical (decrease -16)
    
    // Detailed scaling logic based on bubble rate percentage
    double bubble_rate_percent = bubble_rate * 100.0;
    
    if (bubble_rate_percent >= 40.0) {
        // Very high bubble rate - aggressive reduction
        new_size = current_size - 16;
        printf("[ADAPTIVE] Critical bubble rate (%.2f%% >= 40%%), reducing batch size by 16\n", bubble_rate_percent);
        
    } else if (bubble_rate_percent >= 25.0) {
        // High bubble rate - significant reduction
        new_size = current_size - 8;
        printf("[ADAPTIVE] High bubble rate (%.2f%% >= 25%%), reducing batch size by 8\n", bubble_rate_percent);
        
    } else if (bubble_rate_percent >= 15.0) {
        // Moderate bubble rate - moderate reduction
        new_size = current_size - 4;
        printf("[ADAPTIVE] Moderate bubble rate (%.2f%% >= 15%%), reducing batch size by 4\n", bubble_rate_percent);
        
    } else if (bubble_rate_percent >= 10.0) {
        // Acceptable bubble rate - no change
        new_size = current_size;
        printf("[ADAPTIVE] Acceptable bubble rate (%.2f%% 10-15%%), maintaining batch size %d\n", bubble_rate_percent, current_size);
        
    } else if (bubble_rate_percent >= 5.0) {
        // Low bubble rate - conservative increase
        new_size = current_size + 4;
        printf("[ADAPTIVE] Low bubble rate (%.2f%% 5-10%%), increasing batch size by 4\n", bubble_rate_percent);
        
    } else {
        // Very low bubble rate - aggressive increase
        new_size = current_size + 8;
        printf("[ADAPTIVE] Very low bubble rate (%.2f%% < 5%%), increasing batch size by 8\n", bubble_rate_percent);
    }
    
    // Apply safety bounds
    if (new_size < 16) new_size = 16;
    if (new_size > 128) new_size = 128;
    
    // Prevent oscillation - don't change if difference is small and bubble rate is not critical
    if (abs(new_size - current_size) < 4 && bubble_rate_percent < 25.0) {
        new_size = current_size;
        printf("[ADAPTIVE] Preventing oscillation, maintaining batch size %d\n", current_size);
    }
    
    return new_size;
}

// Advanced bubble rate analysis for pipeline optimization
void analyze_bubble_rate_trend(PipelineConfig* config, PipelineStats* stats) {
    (void)stats; // Suppress unused parameter warning
    
    static double previous_bubble_rates[10] = {0};
    static int bubble_rate_index = 0;
    static int bubble_rate_count = 0;
    
    double current_bubble_rate = config->bubble_rate / 100.0;
    
    // Store current bubble rate
    previous_bubble_rates[bubble_rate_index] = current_bubble_rate;
    bubble_rate_index = (bubble_rate_index + 1) % 10;
    if (bubble_rate_count < 10) bubble_rate_count++;
    
    // Calculate trend if we have enough data
    if (bubble_rate_count >= 5) {
        double sum = 0.0;
        for (int i = 0; i < bubble_rate_count; i++) {
            sum += previous_bubble_rates[i];
        }
        double avg_bubble_rate = sum / bubble_rate_count;
        
        // Check for increasing trend
        int increasing_trend = 0;
        if (bubble_rate_count >= 3) {
            int recent_idx = (bubble_rate_index - 1 + 10) % 10;
            int prev_idx = (bubble_rate_index - 2 + 10) % 10;
            int prev2_idx = (bubble_rate_index - 3 + 10) % 10;
            
            if (previous_bubble_rates[recent_idx] > previous_bubble_rates[prev_idx] &&
                previous_bubble_rates[prev_idx] > previous_bubble_rates[prev2_idx]) {
                increasing_trend = 1;
            }
        }
        
        if (increasing_trend && avg_bubble_rate > 0.02) {
            printf("[ADAPTIVE] ⚠️  Bubble rate trending up (avg: %.2f%%), suggesting pipeline instability\n", 
                   avg_bubble_rate * 100.0);
        }
    }
}

// Adaptive pipeline depth adjustment based on bubble rate
void adjust_pipeline_depth(PipelineConfig* config, PipelineStats* stats) {
    if (!config || !stats) return;
    
    double bubble_rate = config->bubble_rate / 100.0;
    
    // Only adjust if we have significant data
    if (stats->processed_batches < 20) return;
    
    if (bubble_rate > 0.1) { // 10%
        // Very high bubble rate - reduce pipeline depth
        if (config->pipeline_depth > 2) {
            config->pipeline_depth--;
            printf("[ADAPTIVE] High bubble rate, reducing pipeline depth to %d\n", config->pipeline_depth);
        }
    } else if (bubble_rate < 0.01 && config->communication_ratio < 0.2) {
        // Very low bubble rate and good communication - can increase depth
        if (config->pipeline_depth < MAX_PIPELINE_DEPTH) {
            config->pipeline_depth++;
            printf("[ADAPTIVE] Excellent performance, increasing pipeline depth to %d\n", config->pipeline_depth);
        }
    }
} 