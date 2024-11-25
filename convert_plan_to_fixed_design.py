import pandas as pd
from ruamel.yaml import YAML
import re

yaml = YAML()

HARDCODED_LINK_DIRECTIONS = ['NL11,NL12','NL11,NL13',
                             'NL31,NL33','NL32,NL33',
                             'NL33,NL41','NL34,NL41',
                             'NL41,NL42','NL12,NL23',
                             'NL13,NL21','NL21,NL22',
                             'NL21,NL23','NL22,NL41',
                             'NL23,NL32','NL31,NL32']

def plan_results_to_fixedcap_override(results, filename='generic'):
    
    caps = results.get_formatted_array('energy_cap').to_pandas()
    caps_max = results.get_formatted_array('energy_cap_max').to_pandas()
    safe_oversize = 1
    override_dict = {'overrides':{'fix-design-to-{}'.format(filename):{'locations':{}}}}
    all_techs = [x for x in caps.columns if 'transmission' not in x and 
             'demand' not in x and 'lost' not in x and 'curtailment' not in x]
    links = [x for x in caps.columns if 'transmission' in x]
    
    # Fixing generation, storage and conversion caps
    for loc in results.inputs.locs.values:
        override_dict['overrides']['fix-design-to-{}'.format(filename)]['locations'][loc+'.techs'] = {}
        loc_techs = [re.split('([:])',x)[-1] for x in results.inputs.lifetime.to_pandas().index if loc in x]
        techs = [x for x in all_techs if x in loc_techs]
        for tech in techs:
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['locations'][loc+'.techs'][tech] = {}
            override_dict['overrides']['fix-design-to-{}'.format(filename)]['locations'][loc+'.techs'][tech]['constraints'] = {}
            if caps[tech].loc[loc] > 1e-7 and caps[tech].loc[loc] != float('nan'): #filter out noise, i.e. very small numbers
                feasible_cap = min(safe_oversize*caps[tech].loc[loc],caps_max[tech].loc[loc])
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['locations'][loc+'.techs'][tech]['constraints']['energy_cap_equals'] = float(feasible_cap)
            else:
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['locations'][loc+'.techs'][tech]['constraints']['energy_cap_equals'] = float(0)
    
    # Fixing transmission caps
    override_dict['overrides']['fix-design-to-{}'.format(filename)]['links'] = {}
    for loc in results.inputs.locs.values:
        for tech in links:
            end_node = re.split('([:])',tech)[-1]
            pure_tech = re.split('([:])',tech)[0]
            if caps[tech].loc[loc] >= 0 and (loc+',{}'.format(end_node)) in HARDCODED_LINK_DIRECTIONS: # check if NaN
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['links'][loc+',{}.techs'.format(end_node)] = {}
                override_dict['overrides']['fix-design-to-{}'.format(filename)]['links'][
                    loc+',{}.techs'.format(end_node)][pure_tech+'.constraints.energy_cap_equals'] = float(caps[tech].loc[loc])
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
