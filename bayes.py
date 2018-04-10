import numpy as np
import functools

import ratingbase as rb

def rating_func(r1, r2):
    a = r1 * (1.0-r2)
    b = r2 * (1.0-r1)
    return a / (a+b)

def calc_log_prior(r):
    if r>0.0 and r<1.0:
        return 0.0
    else:
        return -np.inf

class TeamNode:
    prop_std = 0.2
    def __init__(self, team):
        self.team = team
        self.win_flags = self.team.games['WL'] == 'W'
        self.rating = 0.5 # Default initialization
        self.nacc = 0

    def set_opponent_pointers(self, team_node_list):
        self.opponents = self.team.games['OPP_IDX'].apply(team_node_list.__getitem__)

    def propose_move(self):
        return np.random.normal(self.rating, self.prop_std)

    def eval_move(self, rating_z):
        pwin_vals = [rating_func(rating_z, o.rating) for o in self.opponents]
        return sum([np.log(p) if win else np.log(1.0-p) for win,p in zip(self.win_flags, pwin_vals)])

    def update_logp(self):
        self.logp = self.eval_move(self.rating)

    def accept_move(self, rating_z, logp_z):
        self.nacc += 1
        self.rating = rating_z
        self.logp = logp_z
        # Tell other teams to update their logp
        for team in self.opponents:
            team.update_logp()

class Bayes(rb.RatingSystm):
    def __init__(self):
        super().__init__()
        self.fit_ratings()

    def fit_ratings(self):
        team_nodes = [TeamNode(team) for team in self.teams]
        for team_node in team_nodes:
            team_node.set_opponent_pointers(team_nodes)
            team_node.update_logp()
            
        self.run_MCMC(team_nodes)
        for team,team_node in zip(self.teams, team_nodes):
            team.rating = team_node.rating
        self.ratings = np.array([t.rating for t in self.teams])

    def run_MCMC(self, team_nodes, nsamp=200, nburn=100):
        # Allocate memory to store samples:
        for team_node in team_nodes:
            team_node.rating_samp = np.empty(nsamp)
            team_node.logp_samp = np.empty(nsamp)

        for isamp in range(nsamp+nburn):
            if isamp%100==0: print(isamp)
            for team_node in team_nodes:
                rating_z = team_node.propose_move()
                logp_z = calc_log_prior(rating_z)
                if np.isfinite(logp_z):
                    logp_z += team_node.eval_move(rating_z)
                if np.exp(logp_z - team_node.logp) > np.random.rand():
                    team_node.accept_move(rating_z, logp_z)

                if isamp >= nburn:
                    team_node.rating_samp[isamp-nburn] = team_node.rating
                    team_node.logp_samp[isamp-nburn] = team_node.logp

        # Compute acceptance rates
        for team_node in team_nodes:
            team_node.acc_ratio = float(team_node.nacc) / (nsamp+nburn)

        # Set ratings to mean
        for team_node in team_nodes:
            team_node.rating = np.mean(team_node.rating_samp)

        # Print acceptance rate stats
        print('Average acceptance rate:', np.mean([t.acc_ratio for t in team_nodes]))
        print('Min acceptance rate:', min([t.acc_ratio for t in team_nodes]))
        print('Max acceptance rate:', max([t.acc_ratio for t in team_nodes]))
        
        
if __name__ == '__main__':
    import pandas as pd

    bayes = Bayes()
    #bayes.evaluate_predicted_wins()
    ratings = pd.DataFrame({'rating':bayes.ratings}, index=[t.name for t in bayes.teams])
    ratings = ratings.sort_values(by='rating', ascending=False)    
