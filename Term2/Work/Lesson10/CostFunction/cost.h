#ifndef COST_H
#define COST_H

double goal_distance_cost(int goal_lane, int intended_lane, int final_lane, 
                          double distance_to_goal);

double inefficiency_cost(int target_speed, int intended_lane, int final_lane, 
                         const std::vector<int> &lane_speeds);

#endif  // COST_H