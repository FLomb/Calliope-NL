import calliope
import xarray as xr
import numpy as np

###---------------------------------------------
# SPORES algorithm
###---------------------------------------------

def run_spores(number_of_spores, cost_optimal_model, scoring_method='integer'):
    # Create some lists to store results as they get generated
    spores = [] # full results
    scores = [] # scores only
    spores_counter = 1
    number_of_spores = number_of_spores
    model = cost_optimal_model

    least_feasible_cost = model.results.cost.loc[{"costs": "monetary"}].sum().sum()
    
    for i in range(spores_counter, spores_counter + number_of_spores):
    
        if spores_counter == 1:
            # Store the cost-optimal results
            spores.append(model.results.expand_dims(spores=[0]))
            scores.append(
                model.backend.get_parameter("cost_flow_cap", as_backend_objs=False)
                .sel(costs="spores_score")
                .expand_dims(spores=[0])
            )
            # Update the slack-cost backend parameter based on the calculated minimum feasible system design cost
            least_feasible_cost
            model.backend.update_parameter("spores_cost_max", least_feasible_cost)
            # Update the objective_cost_weights to reflect the ones defined for the SPORES mode
            model.backend.update_parameter(
                "objective_cost_weights", model.inputs.spores_objective_cost_weights
            )
        else:
            pass
    
        # Calculate weights based on a scoring method
        spores_score = score_via_method(scoring_method, model.results, model.backend)
        # Assign a new score based on the calculated penalties
        model.backend.update_parameter(
            "cost_flow_cap", spores_score.reindex_like(model.inputs.cost_flow_cap)
        )
        # Run the model again to get a solution that reflects the new penalties
        model.solve(force=True)

        results = model.results.expand_dims(spores=[i])

        # Store the results
        spores.append(results)
        scores.append(
            model.backend.get_parameter("cost_flow_cap", as_backend_objs=False)
            .sel(costs="spores_score")
            .expand_dims(spores=[i])
        )
    
        spores_counter += 1
            
    # Concatenate the results in the storage lists into xarray objects 
    spore_ds = xr.concat(spores, dim="spores", combine_attrs="drop")
    score_da = xr.concat(scores, dim="spores")
    backend = model.backend

    return spore_ds, score_da, backend


###--------------------------------------------
# SCORING methods
###--------------------------------------------


def score_via_method(scoring_method, results, backend):

    scoring_method=scoring_method
    results=results
    
    def scoring_integer(results, backend):
        # Filter for technologies of interest
        spores_techs = backend.inputs["spores_tracker"].notnull()
        # Look at capacity deployment in the previous iteration
        previous_cap = results.flow_cap 
        # Make sure that penalties are applied only to non-negligible deployments of capacity
        min_relevant_size = 0.1 * previous_cap.where(spores_techs).max(
            ["nodes", "carriers", "techs"]
        )
        # Where capacity was deployed more than the minimal relevant size, assign an integer penalty (score)
        new_score = previous_cap.copy()
        new_score = new_score.where(spores_techs, other=0)
        new_score = new_score.where(new_score > min_relevant_size, other=0)
        new_score = new_score.where(new_score == 0, other=1000)
        # Transform the score into a "cost" parameter
        new_score.rename("cost_flow_cap")
        new_score = new_score.expand_dims(costs=["spores_score"]).copy()
        new_score = new_score.sum("carriers")
        # Extract the existing cost parameters from the backend
        all_costs = backend.get_parameter("cost_flow_cap", as_backend_objs=False)
        try:
            all_costs = all_costs.expand_dims(nodes=results.nodes).copy()
        except:
            pass
        # Create a new version of the cost parameters by adding up the calculated scores
        new_all_costs = all_costs
        new_all_costs.loc[{"costs":"spores_score"}] += new_score.loc[{"costs":"spores_score"}]
    
        return new_all_costs
    
    def scoring_random(results, backend):
        # Filter for technologies of interest
        spores_techs = backend.inputs["spores_tracker"].notnull()
        # Look at capacity deployment in the previous iteration
        previous_cap = results.flow_cap 
        
        # Assign a random penalty (score)
        new_score = previous_cap.copy()
        new_score = new_score.where(spores_techs, other=0)
        new_score = new_score.where(
            new_score == 0, 
            other=np.random.choice([0,1000],size=(previous_cap.shape)))
        # Transform the score into a "cost" parameter
        new_score.rename("cost_flow_cap")
        new_score = new_score.expand_dims(costs=["spores_score"]).copy()
        new_score = new_score.sum("carriers")
        # Extract the existing cost parameters from the backend
        all_costs = backend.get_parameter("cost_flow_cap", as_backend_objs=False)
        try:
            all_costs = all_costs.expand_dims(nodes=results.nodes).copy()
        except:
            pass
        # Create a new version of the cost parameters by adding up the calculated scores
        new_all_costs = all_costs
        new_all_costs.loc[{"costs":"spores_score"}] += new_score.loc[{"costs":"spores_score"}]
    
        return new_all_costs

    allowed_methods = {
            'integer': scoring_integer,
            # 'relative_deployment': _cap_loc_score_relative_deployment,
            'random':scoring_random,
            # 'evolving_average': _cap_loc_score_evolving_average
            }

    return(allowed_methods[scoring_method](results, backend))



