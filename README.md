# Automated-Voting
Automatic design of voting rules using neural networks and the genetic algorithm. 

=====================================

existing_rules.py: 

```
def borda(profile):
    pass

def plurality(profile):
    pass

def condorcet_exists(profile):
    pass

def get_condorcet_winner(profile):
    pass

def median(profile):
    pass

```
=====================================

generate_data.py:

```
class Profile:
    def __init__(self, num_alternatives, num_voters, distribution):
        self.n_alternatives = num_alternatives
        self.n_voters = num_voters
        self.distr = distribution
        self.profile = np.array(num_alternatives, num_voters)

class PreferenceGenerator:

    def generate_ballot():
        pass

    def generate_profiles(num_profiles):
        pass

def build_dataset(distribution_params, generate=True):
    generator = PreferenceGenerator()
    profiles = generator.generate_profiles(num_profiles)

    return profiles
```
=====================================

genetic.py: 

```
class GeneticAlgorithm:
    def __init__(self):
        pass

    def fitness():
        pass
```
=====================================

nn.py: 

```
class VotingNN:

    def __init__(self, network_params, constraint_list):
        self.network_params = network_params
        self.constraints = constraint_list

    def train():
        pass

    def loss():
        pass

    def predict():
        pass

    def evaluate_constraints():
        pass

```
=====================================

main.py: 

```
def main():
    # 1. Generate or read and process data: Preference profiles as a dataframe/matrix
    dataset = build_dataset("normal", generate=True)

    # 2. Train/test split
    train_X, test_X, train_y, test_y = train_test_split(dataset, t_perc=0.2)

    # 3. Initialize optimizer model
    v_nn = VotingNN()

    # 4. Run/train
    v_nn.train(train_X, train_y)

    # 5. Evaluate
    preds = v_nn.predict(test_X)

```
=====================================
