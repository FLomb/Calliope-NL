import pandas as pd
from ruamel.yaml import YAML
import re

yaml = YAML()

def plan_results_to_fixedcap_override(model, filename='generic'):
    
    results = model.results    
    inputs = model.inputs
    caps = results.flow_cap.sel(carriers='power').to_pandas()
    caps_max = inputs.flow_cap_max.to_pandas()
    safe_oversize = 1
    override_dict = {'overrides':{'fix-design-to-{}'.format(filename):{'nodes':{}, 'techs':{}}}}
    all_techs = [x for x in caps.columns if '_to_' not in x and 
             'demand' not in x and 'lost' not in x and 'curtailment' not in x and
             'export' not in x and 'import' not in x]
    links = [x for x in caps.columns if '_to_' in x]
    
    # Fixing generation, storage and conversion caps
    nl_nodes = [x for x in inputs.nodes.values if 'NL' in x]    
    for loc in nl_nodes:
        override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'] = {}
        techs = [x for x in results.flow_cap.sel(carriers='power').to_series().dropna().loc[loc].index if x in all_techs]
        for tech in techs:
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'][tech] = {}
            if caps[tech].loc[loc] > 0.001 and caps[tech].loc[loc] != float('nan'): #filter out noise, i.e. very small numbers
                feasible_cap = min(safe_oversize*caps[tech].loc[loc],caps_max[tech].loc[loc])
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'][tech]['flow_cap_max'] = round(float(feasible_cap))
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'][tech]['flow_cap_min'] = round(float(feasible_cap))
            else:
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'][tech]['flow_cap_max'] = float(0)
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['nodes'][loc+'.techs'][tech]['flow_cap_min'] = float(0)
    
    # Fixing transmission caps
    for tech in links:
        end_node = re.split('(_to_)',tech)[-1]
        start_node = re.split('(_to_)',tech)[0]
        if caps[tech].loc[start_node] >= 0: 
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech] = {}
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech]['flow_cap_max'] = round(float(caps[tech].loc[start_node]))
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech]['flow_cap_min'] = round(float(caps[tech].loc[start_node]))
        else:
            continue   

    # Convert all into a YAML file
    yaml.default_flow_style = False
    yaml.width=1000
    with open('model_files/scenarios/fixed-design-{}.yaml'.format(filename), 'w') as outfile:
        yaml.dump(override_dict, outfile)
        
    with open("model_files/scenarios/fixed-design-{}.yaml".format(filename), "r+") as f:
        contents = f.read()
        f.seek(0)
        f.write(contents.replace('"', ''))
        f.truncate()
