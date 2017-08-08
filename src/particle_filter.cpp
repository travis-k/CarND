/*
 * particle_filter.cpp
 *
 *      Author: Tiffany Huang, Travis Krebs
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
#include <valarray>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;

	default_random_engine gen;	 

	// This line creates a normal (Gaussian) distribution for x,y,theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	vector<double> weights(num_particles, 0.0);

	for(int i = 0; i < num_particles; i++)
	{				
		Particle par = Particle();
		par.id = i;
		par.x = dist_x(gen);
		par.y = dist_y(gen);
		par.theta = dist_theta(gen);
		par.weight = 1;	

		weights[i] = par.weight;
		
		particles.push_back(par);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Takes in delta time, yaw_rate, velocity and position uncertainty and makes a prediction of where
	// the particle will be at t + 1
	default_random_engine gen;	 

	for(int i = 0; i < num_particles; i++)
	{				
		// Particle position at time t
		double xo = particles[i].x;
		double yo = particles[i].y;
		double thetao = particles[i].theta;

		double theta;
		double x;
		double y;

		// Particle at time t+1 (with crude correction for when yaw_rate is small)
		if (abs(yaw_rate) <= 1e-10) yaw_rate = 1e-10;

		theta = thetao + yaw_rate*delta_t;
		x = xo + (velocity/yaw_rate)*(sin(theta) - sin(thetao));
		y = yo + (velocity/yaw_rate)*(cos(thetao) - cos(theta));
		
		// This line creates a normal (Gaussian) distribution for x,y,theta
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	} 
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Associates an observation with a landmark. It runs quite slowly as-is, slower than two nested for-loops
	// I had hopes the valarray elementwise math would be faster

	// valarray<double> pred_dist (predicted.size());
	// for(int i = 0; i < predicted.size(); i++){
	// 	pred_dist[i] = predicted[i].x*predicted[i].x + predicted[i].y*predicted[i].y;
	// }

	// for (int i = 0; i < observations.size(); ++i) {
	// 	valarray<double> temp ((observations[i].x*observations[i].x + observations[i].y*observations[i].y), predicted.size());	
	// 	temp = abs(temp - pred_dist);
 // 		int min_pos = distance(begin(temp),min_element(begin(temp),end(temp)));
	// 	observations[i].id = min_pos;
	// }

	double odis = 0;
	double pdis = 0;
	double dis; 

	for(int i = 0; i < observations.size(); i++){
		dis = 1e+20;
		odis = observations[i].x*observations[i].x + observations[i].y*observations[i].y;

		for(int ii = 0; ii < predicted.size(); ii++){
			pdis = predicted[ii].x*predicted[ii].x + predicted[ii].y*predicted[ii].y;
			if (abs(odis - pdis) < dis){
				observations[i].id = ii; 
				dis = abs(odis - pdis);
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {

	vector<Particle> pars;
	vector<double> weis;

	double ssensor_range = sensor_range*sensor_range;
	double preamble = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
	double ssx = std_landmark[0]*std_landmark[0];
	double ssy = std_landmark[1]*std_landmark[1];

	double xo = 0.0;
	double yo = 0.0;
	double xm = 0.0;
	double ym = 0.0;
	double o_dis = 0.0;

	vector<LandmarkObs> p_observations;
	vector<LandmarkObs> predicted;
	LandmarkObs objs;
	LandmarkObs tempxy;

	for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
        objs.id = map_landmarks.landmark_list[i].id_i;
		objs.x = (double)map_landmarks.landmark_list[i].x_f;
		objs.y = (double)map_landmarks.landmark_list[i].y_f;
		predicted.push_back(objs);
	}

	for(int i = 0; i < num_particles; i++){
		Particle p = particles[i];
		p_observations.clear();

		for(int ii = 0; ii < observations.size(); ii++){
			xo = observations[ii].x;
			yo = observations[ii].y;

			xm = (p.x + (cos(p.theta)*xo) - (sin(p.theta)*yo));
			ym = (p.y + (sin(p.theta)*xo) + (cos(p.theta)*yo));

			o_dis = (xm - p.x)*(xm - p.x) + (ym - p.y)*(ym - p.y);

			if (o_dis <= ssensor_range){
				objs.x = xm;
				objs.y = ym;
				p_observations.push_back(objs);
			}
		}

		dataAssociation(predicted, p_observations);

		p.weight = 1.0;
		for (int j = 0; j < p_observations.size(); j++) {
			int id = p_observations[j].id;
			tempxy = predicted[p_observations[j].id];

			double num1 = tempxy.x - p_observations[j].x;
			double num2 = tempxy.y - p_observations[j].y;
			double power = -((num1*num1/ssx) + (num2*num2/ssy));

			p.weight *= preamble*exp(power);
		}
		pars.push_back(p);
		weis.push_back(p.weight);
	}

	particles.clear();
	particles = pars;

	weights.clear();
	weights = weis;
}

void ParticleFilter::resample() {
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen; 
	discrete_distribution<> distribution(weights.begin(), weights.end());

	vector<Particle> pars; 
    	for (int i = 0; i < num_particles; ++i) {
       		int idx = distribution(gen);
        	pars.push_back(particles[idx]);
    	}

	particles.clear();
	particles = pars;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
