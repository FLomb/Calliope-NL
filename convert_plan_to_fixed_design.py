import pandas as pd
from ruamel.yaml import YAML
import re

yaml = YAML()

HARDCODED_LINK_DIRECTIONS = ['NL11_to_NL12','NL11_to_NL13',
                             'NL31_to_NL33','NL32_to_NL33',
                             'NL33_to_NL41','NL34_to_NL41',
                             'NL41_to_NL42','NL12_to_NL23',
                             'NL13_to_NL21','NL21_to_NL22',
                             'NL21_to_NL23','NL22_to_NL41',
                             'NL23_to_NL32','NL31_to_NL32',
                             'NL34,BEL','NL42,DEU',
                             'NL12,DNK','NL33,GBR',
                             'NL11,NOR']

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
    for loc in inputs.nodes.values:
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
        end_node = re.split('([_to_])',tech)[-1]
        start_node = re.split('([_to_])',tech)[0]
        if caps[tech].loc[start_node] >= 0: # and (loc+',{}'.format(end_node)) in HARDCODED_LINK_DIRECTIONS: # check if NaN
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech] = {}
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech]['flow_cap_max'] = round(float(caps[tech].loc[start_node]))
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['techs'][tech]['flow_cap_min'] = round(float(caps[tech].loc[start_node]))
        else:
            continue   

    # Convert all into a YAML file
    yaml.default_flow_style = False
    yaml.width=1000
    with open('model_files/fixed-design-{}.yaml'.format(filename), 'w') as outfile:
        yaml.dump(override_dict, outfile)
        
    with open("model_files/fixed-design-{}.yaml".format(filename), "r+") as f:
        contents = f.read()
        f.seek(0)
        f.write(contents.replace('"', ''))
        f.truncate()
