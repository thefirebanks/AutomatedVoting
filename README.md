# Automated-Voting
Automatic design of voting rules using neural networks and the genetic algorithm. 

=====================================

### TODO:
1. Look into manipulation methods in depth (use population subset?)
2. Write the methods for the Condorcet/Majority/IIA/Monotonicity constraints (inherit from Population/Profile)
3. Define the loss function (Done)
4. Figure out the differentiability of the loss function theoretically (Done)
5. Look into writing the genetic algorithm (Done)
=====================================

profiles.py:

```
create_profile_from_distribution(n_voters, candidates, pop_type="spheroid"):

create_profile_from_data()

profile_to_nn_input()

generate_profile_dataset(num_profiles, n_voters, candidates, from_file=False)

```

=====================================

election.py:

```
# 1. Create a profile
profile = create_profile_from_distribution(500, ["Adam", "Bert", "Chad"])
   
# 2. Set weights in order of position
weights = [25, 23, 22]

# 3. Run election
results = election(profile, weights)

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
    dataset = generate_profile_dataset(num_profiles, n_voters, candidates)

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
