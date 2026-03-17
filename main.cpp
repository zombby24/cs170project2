#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std;

//function declarations
void forward_selection(const vector<vector<double>>& data);
void backward_elimination(const vector<vector<double>>& data);
double leave_one_out_cross_validation(const vector<vector<double>>& data, vector<int> current_set_of_features, int k);

int main()
{
    //menu
    string filename;
    cout << "Welcome to Abby Allers' Feature Selection Algorithm!\n"
    << "Type in the name of the file to test: ";
    getline(cin, filename);

    //open file
    ifstream dataFile(filename);
    //dataFile.open(filename);
    if (!dataFile.is_open()) 
    {
        cout << "ERROR: could not open file " << filename << endl;
        return 1;
    }

    //read data
    vector<vector<double>> data;
    double val;
    string line;
    while (getline(dataFile, line)) 
    {
        istringstream iss(line);
        vector<double> row;
        while (iss >> val) 
        {
            row.push_back(val);
        }
        if (!row.empty()) 
        {
            data.push_back(row);
        }
    }

    int selection;
    cout << "Type the number of the algorithm you want to run.\n\n"
    << "1) Forward Selection\n2)Backward Elimination\n";
    cin >> selection;
    switch (selection)
    {
    case 1:
        //forward selection
        forward_selection(data);
        break;
    case 2:
        //backward elim
        backward_elimination(data);
        break;
    default:
        cout << "\nERROR: invalid selection. exiting program...\n";
        break;
    }

    //close file before ending program
    dataFile.close();
    return 0;
}

void forward_selection(const vector<vector<double>>& data) {
    vector<int> current_set_of_features; 
    int num_features = data[0].size() - 1;

    vector<int> best_overall_features;  // track best set across all levels
    double best_overall_accuracy = 0.0;

    for (int i = 1; i <= num_features; ++i) 
    {
        int feature_to_add_at_this_level = -1;
        double best_so_far_accuracy = 0.0;

        for (int k = 1; k <= num_features; ++k) 
        {
            // Check if k is already in current_set_of_features
            auto it = find(current_set_of_features.begin(), current_set_of_features.end(), k);
            if (it == current_set_of_features.end())
            {
                // Test accuracy adding feature k to the set
                double accuracy = leave_one_out_cross_validation(data, current_set_of_features, k);
                
                cout << "Using feature " << k << ", accuracy is: " << accuracy << endl;

                if (accuracy > best_so_far_accuracy) 
                {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        }
        if (feature_to_add_at_this_level != -1) 
        {
            current_set_of_features.push_back(feature_to_add_at_this_level);
            cout << "On level " << i << ", I added feature " << feature_to_add_at_this_level << endl;
            // update best overall 
            if (best_so_far_accuracy > best_overall_accuracy)
            {
                best_overall_accuracy = best_so_far_accuracy;
                best_overall_features = current_set_of_features;
            }
        }
    }
    // print best feature set found
    cout << "Finished search!! The best feature subset is {";
    for (int i = 0; i < best_overall_features.size(); ++i)
    {
        if (i > 0) cout << ",";
        cout << best_overall_features[i];
    }
    cout << "}, which has an accuracy of " << best_overall_accuracy * 100 << "%" << endl;
}

void backward_elimination(const vector<vector<double>>& data)
{
    int num_features = data[0].size() - 1;
    
    // start with ALL features in the set
    vector<int> current_set_of_features;
    for (int i = 1; i <= num_features; ++i)
        current_set_of_features.push_back(i);

    vector<int> best_overall_features = current_set_of_features;
    double best_overall_accuracy = 0.0;

    for (int i = 1; i <= num_features; ++i)
    {
        int feature_to_remove_at_this_level = -1;
        double best_so_far_accuracy = 0.0;

        for (int k = 1; k <= num_features; ++k)
        {
            // only consider features still in the current set
            auto it = find(current_set_of_features.begin(), current_set_of_features.end(), k);
            if (it != current_set_of_features.end())
            {
                // build a temporary set without feature k
                vector<int> temp_set;
                for (int f : current_set_of_features)
                    if (f != k) temp_set.push_back(f);

                // need at least one feature to test
                if (temp_set.empty()) continue;

                double accuracy = leave_one_out_cross_validation(data, temp_set, -1);
                cout << "Removing feature " << k << ", accuracy is: " << accuracy << endl;

                if (accuracy > best_so_far_accuracy)
                {
                    best_so_far_accuracy = accuracy;
                    feature_to_remove_at_this_level = k;
                }
            }
        }
        if (feature_to_remove_at_this_level != -1)
        {
            current_set_of_features.erase(find(current_set_of_features.begin(), current_set_of_features.end(), feature_to_remove_at_this_level));
            cout << "On level " << i << ", I removed feature " << feature_to_remove_at_this_level << endl;

            if (best_so_far_accuracy > best_overall_accuracy)
            {
                best_overall_accuracy = best_so_far_accuracy;
                best_overall_features = current_set_of_features;
            }
        }
    }

    cout << "Finished search!! The best feature subset is {";
    for (int i = 0; i < best_overall_features.size(); ++i)
    {
        if (i > 0) cout << ",";
        cout << best_overall_features[i];
    }
    cout << "}, which has an accuracy of " << best_overall_accuracy * 100 << "%" << endl;
}

double leave_one_out_cross_validation(const vector<vector<double>>& data, vector<int> current_set, int feature_to_add) {
    // 1. Combine the existing features with the new feature we are testing
    vector<int> features_to_use = current_set;
    if (feature_to_add != -1)
    {
        features_to_use.push_back(feature_to_add);
    }
    int number_correctly_classified = 0; 

    // 2. Loop through each row to classify it
    for (int i = 0; i < data.size(); ++i) 
    { 
        double nearest_neighbor_distance = 1e300; // Large number for infinity 
        int nearest_neighbor_label = -1;

        // 3. Compare row 'i' to every other row 'k'
        for (int k = 0; k < data.size(); ++k) 
        {
            if (i == k) continue; // Don't compare a point to itself! 

            double distance = 0.0;
            // 4. Calculate Euclidean distance only on the selected features
            for (int feature_index : features_to_use) 
            {
                // IMPORTANT: In the file, column 0 is the label. 
                // Feature 1 is at index 1, Feature 2 is index 2, etc. 
                double diff = data[i][feature_index] - data[k][feature_index];
                distance += diff * diff;
            }

            if (distance < nearest_neighbor_distance) 
            { 
                nearest_neighbor_distance = distance; 
                nearest_neighbor_label = data[k][0]; // Label is in the first column 
            }
        }

        // 5. Check if the classification was correct
        if (data[i][0] == nearest_neighbor_label) 
        { 
            number_correctly_classified++; 
        }
    }

    // Return accuracy as a fraction 
    return (double)number_correctly_classified / (double)data.size();
}