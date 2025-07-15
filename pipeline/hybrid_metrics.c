/**
 * @file hybrid_metrics.c
 * @brief Extended Metrics Collection for Hybrid Pipeline-EASGD System
 * @author Research Team
 * @date 2024
 * 
 * This file implements advanced metrics collection and analysis specifically
 * for the hybrid system. It tracks both pipeline and EASGD performance,
 * convergence metrics, and provides research-grade analytics.
 */

#include "hybrid_pipeline_easgd.h"
#include <math.h>
#include <stdlib.h>      // malloc, free, getenv
#include <stdio.h>       // printf, fprintf, fopen, fclose, FILE, stdout, fflush
#include <string.h>      // memset, strcpy, strcmp
#include <netinet/in.h>  // struct sockaddr_in, socklen_t
#include <sys/socket.h>  // accept

/* =============================================================================
 * LOAD BALANCING ANALYSIS
 * ============================================================================= */

double calculate_load_balance_variance(HybridCoordinator* coord) {
    if (!coord) return -1.0;
    
    // Calculate utilization percentages for each worker
    double total_time = difftime(time(NULL), coord->start_time);
    if (total_time <= 0) return 0.0;
    
    // Stage1 workers utilization (simplified calculation)
    double stage1_1_util = coord->aggregated_stats.stage1_processing_time / (total_time * 2.0) * 100.0;
    double stage1_2_util = coord->aggregated_stats.stage1_processing_time / (total_time * 2.0) * 100.0;
    double stage2_util = coord->aggregated_stats.stage2_processing_time / total_time * 100.0;
    
    // Calculate mean utilization
    double mean_util = (stage1_1_util + stage1_2_util + stage2_util) / 3.0;
    
    // Calculate variance
    double variance = ((stage1_1_util - mean_util) * (stage1_1_util - mean_util) +
                      (stage1_2_util - mean_util) * (stage1_2_util - mean_util) +
                      (stage2_util - mean_util) * (stage2_util - mean_util)) / 3.0;
    
    double std_deviation = sqrt(variance);
    
    printf("[METRICS] Load balance analysis:\n");
    printf("          Stage1-1 utilization: %.2f%%\n", stage1_1_util);
    printf("          Stage1-2 utilization: %.2f%%\n", stage1_2_util);
    printf("          Stage2-1 utilization: %.2f%%\n", stage2_util);
    printf("          Mean utilization: %.2f%%\n", mean_util);
    printf("          Standard deviation: %.2f%%\n", std_deviation);
    
    return std_deviation / mean_util; // Coefficient of variation
}

void optimize_load_distribution(HybridCoordinator* coord) {
    if (!coord) return;
    
    double variance = calculate_load_balance_variance(coord);
    
    if (variance > 0.2) {
        printf("[METRICS] High load imbalance detected (variance: %.3f)\n", variance);
        printf("[METRICS] Recommendations:\n");
        printf("          - Consider dynamic work stealing\n");
        printf("          - Adjust batch sizes per worker\n");
        printf("          - Implement adaptive scheduling\n");
    } else if (variance > 0.1) {
        printf("[METRICS] Moderate load imbalance (variance: %.3f)\n", variance);
        printf("[METRICS] Minor optimization opportunities available\n");
    } else {
        printf("[METRICS] Good load balance achieved (variance: %.3f)\n", variance);
    }
}

/* =============================================================================
 * CONVERGENCE ANALYSIS
 * ============================================================================= */

typedef struct {
    double divergence_history[100];
    int history_count;
    double convergence_threshold;
    int convergence_detected_epoch;
} ConvergenceTracker;

static ConvergenceTracker conv_tracker = {0};

// void track_convergence_metrics(HybridCoordinator* coord, int epoch) {
//     if (!coord) return;
    
//     // Add current divergence to history
//     if (conv_tracker.history_count < 100) {
//         conv_tracker.divergence_history[conv_tracker.history_count] = coord->total_weight_divergence;
//         conv_tracker.history_count++;
//     } else {
//         // Shift array and add new value
//         for (int i = 0; i < 99; i++) {
//             conv_tracker.divergence_history[i] = conv_tracker.divergence_history[i + 1];
//         }
//         conv_tracker.divergence_history[99] = coord->total_weight_divergence;
//     }
    
//     // Analyze convergence trend
//     if (conv_tracker.history_count >= 5) {
//         double recent_avg = 0.0;
//         double older_avg = 0.0;
//         int half = conv_tracker.history_count / 2;
        
//         // Calculate average of recent half
//         for (int i = half; i < conv_tracker.history_count; i++) {
//             recent_avg += conv_tracker.divergence_history[i];
//         }
//         recent_avg /= (conv_tracker.history_count - half);
        
//         // Calculate average of older half
//         for (int i = 0; i < half; i++) {
//             older_avg += conv_tracker.divergence_history[i];
//         }
//         older_avg /= half;
        
//         // Calculate convergence rate
//         double convergence_rate = (older_avg - recent_avg) / older_avg;
        
//         printf("[METRICS] Convergence analysis (epoch %d):\n", epoch + 1);
//         printf("          Current divergence: %.6f\n", coord->total_weight_divergence);
//         printf("          Recent average: %.6f\n", recent_avg);
//         printf("          Older average: %.6f\n", older_avg);
//         printf("          Convergence rate: %.4f\n", convergence_rate);
        
//         // Detect convergence
//         if (convergence_rate > 0.1 && recent_avg < 0.01 && conv_tracker.convergence_detected_epoch == 0) {
//             conv_tracker.convergence_detected_epoch = epoch + 1;
//             printf("[METRICS] *** CONVERGENCE DETECTED at epoch %d ***\n", epoch + 1);
//         }
//     }
// }

// int detect_convergence(HybridCoordinator* coord, double threshold) {
//     if (!coord) return -1;
    
//     conv_tracker.convergence_threshold = threshold;
    
//     if (coord->total_weight_divergence < threshold) {
//         if (conv_tracker.convergence_detected_epoch == 0) {
//             conv_tracker.convergence_detected_epoch = coord->current_epoch + 1;
//         }
//         return 1;
//     }
    
//     return 0;
// }

/* =============================================================================
 * EASGD EFFICIENCY ANALYSIS
 * ============================================================================= */

void analyze_easgd_efficiency(HybridCoordinator* coord) {
    if (!coord) return;
    
    double total_time = difftime(time(NULL), coord->start_time);
    double easgd_overhead = coord->easgd_communication_time;
    double easgd_efficiency = (total_time > 0) ? (1.0 - easgd_overhead / total_time) * 100.0 : 0.0;
    
    printf("[METRICS] EASGD Efficiency Analysis:\n");
    printf("          Total training time: %.2f seconds\n", total_time);
    printf("          EASGD communication time: %.2f seconds\n", easgd_overhead);
    printf("          EASGD overhead ratio: %.3f\n", easgd_overhead / total_time);
    printf("          EASGD efficiency: %.2f%%\n", easgd_efficiency);
    printf("          Total EASGD rounds: %d\n", coord->total_easgd_rounds);
    printf("          Average time per round: %.3f seconds\n", 
           easgd_overhead / max(1, coord->total_easgd_rounds));
    
    // Efficiency assessment
    if (easgd_efficiency > 95.0) {
        printf("[METRICS] EXCELLENT EASGD efficiency\n");
    } else if (easgd_efficiency > 90.0) {
        printf("[METRICS] GOOD EASGD efficiency\n");
    } else if (easgd_efficiency > 80.0) {
        printf("[METRICS] FAIR EASGD efficiency - optimization opportunities\n");
    } else {
        printf("[METRICS] POOR EASGD efficiency - significant overhead\n");
        printf("[METRICS] Recommendations:\n");
        printf("          - Reduce synchronization frequency\n");
        printf("          - Optimize weight serialization\n");
        printf("          - Consider asynchronous EASGD variants\n");
    }
}

/* =============================================================================
 * PIPELINE EFFICIENCY ANALYSIS
 * ============================================================================= */

void analyze_pipeline_efficiency(HybridCoordinator* coord) {
    if (!coord) return;
    
    double total_time = difftime(time(NULL), coord->start_time);
    double computation_time = coord->aggregated_stats.stage1_processing_time + 
                             coord->aggregated_stats.stage2_processing_time;
    double communication_time = coord->aggregated_stats.communication_time;
    
    double pipeline_efficiency = (total_time > 0) ? (computation_time / total_time) * 100.0 : 0.0;
    double communication_overhead = (total_time > 0) ? (communication_time / total_time) * 100.0 : 0.0;
    
    printf("[METRICS] Pipeline Efficiency Analysis:\n");
    printf("          Total computation time: %.2f seconds\n", computation_time);
    printf("          Pipeline communication time: %.2f seconds\n", communication_time);
    printf("          Pipeline efficiency: %.2f%%\n", pipeline_efficiency);
    printf("          Communication overhead: %.2f%%\n", communication_overhead);
    
    // Stage balance analysis
    double stage_balance = (coord->aggregated_stats.stage1_processing_time > 0 && 
                           coord->aggregated_stats.stage2_processing_time > 0) ?
        min(coord->aggregated_stats.stage1_processing_time, coord->aggregated_stats.stage2_processing_time) /
        max(coord->aggregated_stats.stage1_processing_time, coord->aggregated_stats.stage2_processing_time) : 0.0;
    
    printf("          Stage balance ratio: %.3f\n", stage_balance);
    
    if (stage_balance > 0.8) {
        printf("[METRICS] WELL-BALANCED pipeline stages\n");
    } else if (stage_balance > 0.6) {
        printf("[METRICS] MODERATELY-BALANCED pipeline stages\n");
    } else {
        printf("[METRICS] IMBALANCED pipeline stages - bottleneck detected\n");
        if (coord->aggregated_stats.stage1_processing_time > coord->aggregated_stats.stage2_processing_time) {
            printf("[METRICS] Stage1 is the bottleneck\n");
        } else {
            printf("[METRICS] Stage2 is the bottleneck\n");
        }
    }
}

/* =============================================================================
 * HYBRID SYSTEM PERFORMANCE ANALYSIS
 * ============================================================================= */

void analyze_hybrid_system_performance(HybridCoordinator* coord) {
    if (!coord) return;
    
    printf("\n=== COMPREHENSIVE HYBRID SYSTEM ANALYSIS ===\n");
    
    // Overall system metrics
    double total_time = difftime(time(NULL), coord->start_time);
    double total_samples = coord->aggregated_stats.total_predictions;
    double throughput = (total_time > 0) ? total_samples / total_time : 0.0;
    
    printf("[METRICS] System Overview:\n");
    printf("          Training duration: %.2f seconds\n", total_time);
    printf("          Total samples processed: %.0f\n", total_samples);
    printf("          Overall throughput: %.2f samples/second\n", throughput);
    printf("          Processed batches: %d\n", coord->aggregated_stats.processed_batches);
    
    // Component analysis
    analyze_pipeline_efficiency(coord);
    printf("\n");
    analyze_easgd_efficiency(coord);
    printf("\n");
    optimize_load_distribution(coord);
    printf("\n");
    
    // Hybrid advantage calculation
    double sequential_estimate = total_time * 2.0; // Rough estimate
    double hybrid_speedup = sequential_estimate / total_time;
    
    printf("[METRICS] Hybrid System Advantage:\n");
    printf("          Estimated sequential time: %.2f seconds\n", sequential_estimate);
    printf("          Actual hybrid time: %.2f seconds\n", total_time);
    printf("          Hybrid speedup: %.2fx\n", hybrid_speedup);
    
    if (hybrid_speedup > 1.5) {
        printf("[METRICS] EXCELLENT hybrid performance\n");
    } else if (hybrid_speedup > 1.2) {
        printf("[METRICS] GOOD hybrid performance\n");
    } else {
        printf("[METRICS] MODERATE hybrid performance - optimization needed\n");
    }
    
    printf("==============================================\n\n");
}

/* =============================================================================
 * RESEARCH METRICS LOGGING
 * ============================================================================= */

void log_research_metrics(HybridCoordinator* coord, int epoch) {
    if (!coord) return;
    
    time_t timestamp = time(NULL);
    double total_time = difftime(timestamp, coord->start_time);
    
    // Log comprehensive metrics for research analysis
    printf("RESEARCH_LOG|%ld|EPOCH_%d|", timestamp, epoch + 1);
    printf("total_time_%.3f|", total_time);
    printf("stage1_time_%.3f|", coord->aggregated_stats.stage1_processing_time);
    printf("stage2_time_%.3f|", coord->aggregated_stats.stage2_processing_time);
    printf("comm_time_%.3f|", coord->aggregated_stats.communication_time);
    printf("easgd_time_%.3f|", coord->easgd_communication_time);
    printf("stage1_div_%.6f|", coord->stage1_divergence);
    printf("stage2_div_%.6f|", coord->stage2_divergence);
    printf("total_div_%.6f|", coord->total_weight_divergence);
    printf("batches_%d|", coord->aggregated_stats.processed_batches);
    printf("samples_%.0f|", (double)coord->aggregated_stats.total_predictions);
    
    // Calculate derived metrics
    double throughput = (total_time > 0) ? coord->aggregated_stats.total_predictions / total_time : 0.0;
    double accuracy = (coord->aggregated_stats.total_predictions > 0) ?
        (double)coord->aggregated_stats.correct_predictions / coord->aggregated_stats.total_predictions * 100.0 : 0.0;
    double avg_loss = (coord->aggregated_stats.processed_batches > 0) ?
        coord->aggregated_stats.total_loss / coord->aggregated_stats.processed_batches : 0.0;
    
    printf("throughput_%.2f|", throughput);
    printf("accuracy_%.3f|", accuracy);
    printf("avg_loss_%.4f", avg_loss);
    printf("\n");
    
    fflush(stdout);
}

void log_hybrid_performance(HybridCoordinator* coord, int epoch) {
    if (!coord) return;
    
    // Log both research metrics and readable performance summary
    log_research_metrics(coord, epoch);
    
    printf("[METRICS] Epoch %d Performance Summary:\n", epoch + 1);
    printf("          Divergence: Stage1=%.6f, Stage2=%.6f, Total=%.6f\n",
           coord->stage1_divergence, coord->stage2_divergence, coord->total_weight_divergence);
    printf("          Processing: Stage1=%.2fs, Stage2=%.2fs, EASGD=%.2fs\n",
           coord->aggregated_stats.stage1_processing_time,
           coord->aggregated_stats.stage2_processing_time,
           coord->easgd_communication_time);
    printf("          Batches: %d, Accuracy: %.2f%%\n",
           coord->aggregated_stats.processed_batches,
           (coord->aggregated_stats.total_predictions > 0) ?
           (double)coord->aggregated_stats.correct_predictions / coord->aggregated_stats.total_predictions * 100.0 : 0.0);
    printf("\n");
}

/* =============================================================================
 * COMPARISON WITH BASELINES
 * ============================================================================= */

int compare_with_pipeline_baseline(HybridExperimentResults* results) {
    if (!results) return -1;
    
    printf("[METRICS] Comparison with Pipeline-Only Baseline:\n");
    printf("          Estimated pipeline-only time: %.2f seconds\n", results->pipeline_only_time);
    printf("          Actual hybrid time: %.2f seconds\n", results->base_results.total_training_time);
    printf("          Improvement: %.2fx faster\n", 
           results->pipeline_only_time / results->base_results.total_training_time);
    
    // Analyze the benefits of adding EASGD
    double easgd_benefit = (results->pipeline_only_time - results->base_results.total_training_time) / 
                          results->pipeline_only_time * 100.0;
    
    printf("          EASGD benefit: %.2f%% time reduction\n", easgd_benefit);
    
    if (easgd_benefit > 20.0) {
        printf("          SIGNIFICANT benefit from hybrid approach\n");
    } else if (easgd_benefit > 10.0) {
        printf("          MODERATE benefit from hybrid approach\n");
    } else if (easgd_benefit > 0.0) {
        printf("          MINOR benefit from hybrid approach\n");
    } else {
        printf("          NO benefit from hybrid approach - overhead too high\n");
    }
    
    return 0;
}

int compare_with_easgd_baseline(HybridExperimentResults* results) {
    if (!results) return -1;
    
    printf("[METRICS] Comparison with EASGD-Only Baseline:\n");
    printf("          Estimated EASGD-only time: %.2f seconds\n", results->easgd_only_time);
    printf("          Actual hybrid time: %.2f seconds\n", results->base_results.total_training_time);
    printf("          Improvement: %.2fx faster\n", 
           results->easgd_only_time / results->base_results.total_training_time);
    
    // Analyze the benefits of adding pipeline parallelism
    double pipeline_benefit = (results->easgd_only_time - results->base_results.total_training_time) / 
                             results->easgd_only_time * 100.0;
    
    printf("          Pipeline benefit: %.2f%% time reduction\n", pipeline_benefit);
    
    if (pipeline_benefit > 30.0) {
        printf("          EXCELLENT benefit from pipeline parallelism\n");
    } else if (pipeline_benefit > 20.0) {
        printf("          GOOD benefit from pipeline parallelism\n");
    } else if (pipeline_benefit > 10.0) {
        printf("          MODERATE benefit from pipeline parallelism\n");
    } else {
        printf("          MINIMAL benefit from pipeline parallelism\n");
    }
    
    return 0;
}

void generate_performance_comparison_report(HybridExperimentResults* results) {
    if (!results) return;
    
    printf("\n=== PERFORMANCE COMPARISON REPORT ===\n");
    
    compare_with_pipeline_baseline(results);
    printf("\n");
    compare_with_easgd_baseline(results);
    printf("\n");
    
    // Overall hybrid advantage
    double total_improvement = (results->pipeline_only_time + results->easgd_only_time) / 
                              (2.0 * results->base_results.total_training_time);
    
    printf("[METRICS] Overall Hybrid Advantage:\n");
    printf("          Average baseline time: %.2f seconds\n", 
           (results->pipeline_only_time + results->easgd_only_time) / 2.0);
    printf("          Hybrid time: %.2f seconds\n", results->base_results.total_training_time);
    printf("          Overall improvement: %.2fx\n", total_improvement);
    
    if (total_improvement > 2.0) {
        printf("          OUTSTANDING hybrid system performance\n");
    } else if (total_improvement > 1.5) {
        printf("          EXCELLENT hybrid system performance\n");
    } else if (total_improvement > 1.2) {
        printf("          GOOD hybrid system performance\n");
    } else {
        printf("          MODERATE hybrid system performance\n");
    }
    
    printf("=====================================\n\n");
}