/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Number of particles to draw
	num_particles = 100;

	// Set Normal Distribution for each parameter
	random_device rd;
	default_random_engine generator(rd());
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize particle and set to particles vector
	for (int i = 0; i < num_particles; ++i) {
		// create a sample particle
		double sample_x, sample_y, sample_theta;
		sample_x     = dist_x(generator);
		sample_y     = dist_y(generator);
		sample_theta = dist_theta(generator);

		// set initial parameters
		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < particles.size(); ++i) {

		// Calculate next position
		double next_x, next_y, next_theta;
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		if (yaw_rate == 0) {
			next_x = x + (velocity * cos(theta) * delta_t);
			next_y = y + (velocity * sin(theta) * delta_t);
			next_theta = theta;
		} else {
			next_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
			next_y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
			next_theta = theta + yaw_rate * delta_t;
		}

		// Add random Gaussian noise
		random_device rd;
		default_random_engine generator(rd());
		normal_distribution<double> dist_x(next_x, std_pos[0]);
		normal_distribution<double> dist_y(next_y, std_pos[1]);
		normal_distribution<double> dist_theta(next_theta, std_pos[2]);

		// Update particle
		particles[i].x     = dist_x(generator);
		particles[i].y     = dist_y(generator);
		particles[i].theta = dist_theta(generator);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];

		// transform observations to map coordinate
		std::vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs = observations[j];
			LandmarkObs transformed_obs;

			transformed_obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
			transformed_obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;

			transformed_observations.push_back(transformed_obs);
		}

		// predict mesurements
		std::vector<LandmarkObs> predictions;
		for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
			if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
				LandmarkObs pred = {landmark.id_i, landmark.x_f, landmark.y_f};
				predictions.push_back(pred);
			}
		}

		// update particle weight
		particle.weight = 1;
		for (int k = 0; k < transformed_observations.size(); ++k) {
			LandmarkObs obs = transformed_observations[k];
			LandmarkObs pred = predictions[obs.id];

			double dx2 = pow(obs.x - pred.x, 2);
			double dy2 = pow(obs.y - pred.y, 2);
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double denom = 1 / (2 * M_PI * std_x * std_y);
			double weight = exp(-((dx2/2*std_x*std_x) + (dy2/2*std_y*std_y)));

			particle.weight *= weight;
		}
		weights[i] = particle.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	random_device rd;
	default_random_engine generator(rd());

	discrete_distribution<int> dist_w(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for (Particle particle : particles) {
		resample_particles.push_back(particles[dist_w(generator)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
