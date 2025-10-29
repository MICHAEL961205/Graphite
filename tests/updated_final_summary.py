import pandas as pd
import os
import numpy as np

def load_results(filename):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(f"evaluation_results/{filename}")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def main():
    print("=== UPDATED TSP SOLVER EVALUATION SUMMARY ===\n")
    
    # List of all algorithms tested
    algorithms = [
        ("TwoOptSolver", "2-opt local search"),
        ("ThreeOptSolver", "3-opt local search"),
        ("SimulatedAnnealingSolver", "Simulated Annealing"),
        ("GeneticAlgorithmSolver", "Basic Genetic Algorithm"),
        ("CheapestInsertionTwoOptSolver", "Cheapest Insertion + 2-opt"),
        ("ChristofidesSolver", "Christofides approximation"),
        ("NearestInsertionTwoOptSolver", "Nearest Insertion + 2-opt"),
        ("FarthestInsertionTwoOptSolver", "Farthest Insertion + 2-opt"),
        ("LinKernighanSolver", "Lin-Kernighan heuristic"),
        ("AdvancedGeneticSolver", "Advanced Genetic Algorithm"),
        ("AdvancedSimulatedAnnealingSolver", "Advanced Simulated Annealing"),
        ("AntColonySolver", "Ant Colony Optimization"),
        ("VariableNeighborhoodSolver", "Variable Neighborhood Search"),
        ("MemeticSolver", "Memetic Algorithm (GA + Local Search)"),
        ("HybridGeneticSolver", "Hybrid Genetic Algorithm")
    ]
    
    results = []
    
    for algo_name, description in algorithms:
        scores_file = f"{algo_name.lower()}_scores.csv"
        times_file = f"{algo_name.lower()}_run_times.csv"
        
        scores_df = load_results(scores_file)
        times_df = load_results(times_file)
        
        if scores_df is not None and times_df is not None:
            # Get algorithm and NN scores
            if algo_name in scores_df.columns and 'NearestNeighbourSolver' in scores_df.columns:
                algo_scores = scores_df[algo_name].values
                nn_scores = scores_df['NearestNeighbourSolver'].values
                
                # Calculate improvements
                improvements = sum(1 for a, n in zip(algo_scores, nn_scores) 
                                 if a != float('inf') and n != float('inf') and a < n)
                comparable = sum(1 for a, n in zip(algo_scores, nn_scores) 
                               if a != float('inf') and n != float('inf'))
                
                # Calculate averages
                avg_algo = np.mean([s for s in algo_scores if s != float('inf')])
                avg_nn = np.mean([s for s in nn_scores if s != float('inf')])
                
                # Calculate average run time
                avg_time = np.mean([t for t in times_df[algo_name].values if t != float('inf')])
                
                # Determine status
                if improvements > 0:
                    status = "✅ BETTER"
                else:
                    status = "❌ NOT BETTER"
                
                results.append({
                    'Algorithm': algo_name,
                    'Description': description,
                    'Improvements': f"{improvements}/{comparable}",
                    'Avg Score': f"{avg_algo:.0f}",
                    'Avg NN Score': f"{avg_nn:.0f}",
                    'Avg Time (s)': f"{avg_time:.1f}",
                    'Status': status
                })
            else:
                results.append({
                    'Algorithm': algo_name,
                    'Description': description,
                    'Improvements': "N/A",
                    'Avg Score': "N/A",
                    'Avg NN Score': "N/A",
                    'Avg Time (s)': "N/A",
                    'Status': "❌ NO DATA"
                })
        else:
            results.append({
                'Algorithm': algo_name,
                'Description': description,
                'Improvements': "N/A",
                'Avg Score': "N/A",
                'Avg NN Score': "N/A",
                'Avg Time (s)': "N/A",
                'Status': "❌ NO DATA"
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save to CSV
    summary_df.to_csv("evaluation_results/updated_algorithm_summary.csv", index=False)
    
    # Print summary
    print("ALGORITHM PERFORMANCE SUMMARY:")
    print("=" * 100)
    print(f"{'Algorithm':<25} {'Improvements':<12} {'Avg Score':<12} {'Avg NN':<12} {'Time(s)':<10} {'Status'}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Algorithm']:<25} {row['Improvements']:<12} {row['Avg Score']:<12} {row['Avg NN Score']:<12} {row['Avg Time (s)']:<10} {row['Status']}")
    
    # Count successful algorithms
    successful = summary_df[summary_df['Status'] == '✅ BETTER']
    print(f"\nSUCCESSFUL ALGORITHMS: {len(successful)}/{len(summary_df)}")
    
    if len(successful) > 0:
        print("\nBEST PERFORMING ALGORITHMS:")
        for _, row in successful.iterrows():
            print(f"  • {row['Algorithm']}: {row['Improvements']} improvements, avg score {row['Avg Score']}")
    
    print(f"\nSummary saved to: evaluation_results/updated_algorithm_summary.csv")

if __name__ == "__main__":
    main()
