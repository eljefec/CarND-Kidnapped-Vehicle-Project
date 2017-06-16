/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

typedef normal_distribution<double> Distribution;

vector<Distribution> MakeDistributions(const double initials[], const double std[])
{
    static const int c_paramCount = 3;

    vector<Distribution> distributions;

    for (int i = 0; i < c_paramCount; i++)
    {
        const double initial = initials[i];
        const double sigma = std[i];
        distributions.emplace_back(Distribution(initial, sigma));
    }

    return distributions;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 20;

    double initials[] = { x, y, theta };

    auto distributions = MakeDistributions(initials, std);

    for (int id = 0; id < num_particles; id++)
    {
        particles.emplace_back(
            Particle { id,
                       distributions[0](m_random_engine),
                       distributions[1](m_random_engine),
                       distributions[2](m_random_engine),
                       1,
                       {},
                       {},
                       {} });
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_yaw = std_pos[2];

    if (yaw_rate == 0)
    {
        for (Particle& p : particles)
        {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);
            // p.theta is unchanged.
        }
    }
    else
    {
        for (Particle& p : particles)
        {
            const double next_yaw = p.theta + yaw_rate * delta_t;

            p.x = p.x + velocity / yaw_rate * (sin(next_yaw) - sin(p.theta));
            p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(next_yaw));
            p.theta = next_yaw;
        }
    }

    // Add random Gaussian noise.
    double initials[] = { 0, 0, 0 };

    auto distributions = MakeDistributions(initials, std_pos);

    for (Particle& p : particles)
    {
        p.x += distributions[0](m_random_engine);
        p.y += distributions[1](m_random_engine);
        p.theta += distributions[2](m_random_engine);
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predictions, std::vector<LandmarkObs>& observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto& observation : observations)
    {
        int closest_id = -1;
        double closest_distance = std::numeric_limits<double>::max();

        for (const auto& prediction : predictions)
        {
            double dist = observation.distance(prediction);

            if (dist < closest_distance)
            {
                closest_id = prediction.id;
                closest_distance = dist;
            }
        }

        observation.id = closest_id;
    }
}

LandmarkObs TransformObservationToMapCoordinates(const LandmarkObs& obs, const Particle& p)
{
    // This is based on equation 3.33 from http://planning.cs.uiuc.edu/node99.html

    double costheta = cos(p.theta);
    double sintheta = sin(p.theta);

    double x = (obs.x * costheta) - (obs.y * sintheta) + p.x;
    double y = (obs.x * sintheta) + (obs.y * costheta) + p.y;

    return LandmarkObs { -1, x, y };
}

double MultivariateGaussianProbability(const LandmarkObs& obs, const LandmarkObs& landmark, const double std_landmark[])
{
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];

    double exponent = -((obs.x - landmark.x) * (obs.x - landmark.x) / (2 * std_x * std_x)
                      + (obs.y - landmark.y) * (obs.y - landmark.y) / (2 * std_y * std_y));

    double denominator = 2 * M_PI * std_x * std_y;

    return (1 / denominator) * exp(exponent);
}

std::vector<LandmarkObs> MakeLandmarks(const Map& map_landmarks)
{
    std::vector<LandmarkObs> landmarks;

    for (const auto& landmark : map_landmarks.landmark_list)
    {
        landmarks.emplace_back( LandmarkObs { landmark.id_i, landmark.x_f, landmark.y_f } );
    }

    return landmarks;
}

std::unordered_map<int, const LandmarkObs*> MakeLandmarkIdMap(const std::vector<LandmarkObs>& landmarks)
{
    std::unordered_map<int, const LandmarkObs*> landmarkIdMap;

    for (const auto& landmark : landmarks)
    {
        landmarkIdMap[landmark.id] = &landmark;
    }

    return landmarkIdMap;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    const auto c_landmarks = MakeLandmarks(map_landmarks);
    const auto c_landmarkIdMap = MakeLandmarkIdMap(c_landmarks);

    for (Particle& p : particles)
    {
        // Transform observation by the particle's position and orientation
        std::vector<LandmarkObs> transformed_observations;
        for (LandmarkObs& obs : observations)
        {
            transformed_observations.emplace_back(TransformObservationToMapCoordinates(obs, p));
        }

        // Associate observation with closest landmark.
        dataAssociation(c_landmarks, transformed_observations);

        // Estimate probability of correspondence between each observation and its associated landmark.
        double finalWeight = 1.0;

        for (LandmarkObs& obs : transformed_observations)
        {
            auto it = c_landmarkIdMap.find(obs.id);

            if (it == c_landmarkIdMap.end())
            {
                throw std::runtime_error("Landmark not found. Fatal error.");
            }

            const LandmarkObs& landmark = *(it->second);

            double prob = MultivariateGaussianProbability(obs, landmark, std_landmark);

            finalWeight *= prob;
        }

        // Update weight.
        p.weight = finalWeight;
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<Particle> resampled;
    std::vector<double> weights;
    for (const auto& p : particles)
    {
        weights.emplace_back(p.weight);
    }

    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; ++i)
    {
        resampled.push_back(particles[distribution(m_random_engine)]);
    }

    particles = resampled;
}

void Particle::SetAssociations(const std::vector<int>& new_associations, const std::vector<double>& new_sense_x, const std::vector<double>& new_sense_y)
{
    // new_associations: The landmark id that goes along with each listed association
    // new_sense_x: the associations x mapping already converted to world coordinates
    // new_sense_y: the associations y mapping already converted to world coordinates

    associations = new_associations;
    sense_x = new_sense_x;
    sense_y = new_sense_y;
}

std::ostream& operator<<(std::ostream& os, const Particle& p)
{
    os << "id=" << p.id << ",(" << p.x << ',' << p.y << "),theta=" << p.theta << ",w=" << p.weight;
}

string ParticleFilter::getAssociations(const Particle& best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(const Particle& best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(const Particle& best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
