�HDF

                    ����������     ��������        `                               OHDR�"     �       jk     �     <�     
          S      �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  j1�nFRHP              �      �m      1       �              P             -:                                           (  o�      t��|BTHD       d(�              ��\MBTHD 	      d(�              h��CFSHD  �                            P x (        ��     D       D       �q�vBTLF  8   6     $� �   P     �*% Z   K     l3�- �    P     �J7 @   �
    kW�G     7      @�k          �T�v "   8 	   ���          Ûŀ    r 
   �2� �    ;     �� M    `     ���� n   6     ʻ#� �   .     ��N��                                                                                                                                                                                                                                                                        BTLF 	     7       M    `      �    P      �    ;      8   6      n   6      �   .      @   �
     �   P      "   8 	       r 
                        Z   K     ͬ�                                                                                                                                                                                                                                                                                                                                BTHD       d(��             ��    _model_run         ͞        scenario: res_3h
applied_overrides: res_3h
techs:
  battery:
    inheritance:
    - storage
    essentials:
      parent: storage
      carrier: electricity
      color: '#33F6FF'
      name: Battery storage
      carrier_in: electricity
      carrier_out: electricity
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - charge_rate
    - energy_cap_per_storage_cap_min
    - energy_cap_per_storage_cap_max
    - energy_cap_per_storage_cap_equals
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_con
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_asynchronous_prod_con
    - lifetime
    - storage_cap_equals
    - storage_cap_max
    - storage_cap_min
    - storage_cap_per_unit
    - storage_initial
    - storage_loss
    - storage_time_max
    - storage_discharge_depth
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - storage_cap_min
    - storage_cap_max
    - storage_cap_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_prod
    - purchase
    - storage_cap
  ccgt:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#AB2C18'
      name: Combined cycle gas turbine
      carrier_in: resource
    constraints:
      energy_cap_max_systemwide: 10000
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  curtailment:
    inheritance:
    - demand
    essentials:
      parent: demand
      carrier_in: electricity
      color: '#6F6C7C'
      name: Electricity curtailment
      carrier_out: resource
    constraints: {}
    required_constraints:
    - resource
    allowed_constraints:
    - energy_con
    - force_resource
    - resource
    - resource_area_equals
    - resource_scale
    - resource_unit
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - carrier_con_min
    - carrier_con_max
    - carrier_con_equals
    allowed_costs:
    - om_con
  demand_electricity:
    inheritance:
    - demand
    essentials:
      parent: demand
      carrier: electricity
      color: '#072486'
      name: Electricity demand
      carrier_in: electricity
      carrier_out: resource
    constraints: {}
    required_constraints:
    - resource
    allowed_constraints:
    - energy_con
    - force_resource
    - resource
    - resource_area_equals
    - resource_scale
    - resource_unit
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - carrier_con_min
    - carrier_con_max
    - carrier_con_equals
    allowed_costs:
    - om_con
  export_electricity:
    inheritance:
    - demand
    essentials:
      parent: demand
      carrier_in: electricity
      color: '#D800FF'
      name: Electricity export
      carrier_out: resource
    constraints: {}
    required_constraints:
    - resource
    allowed_constraints:
    - energy_con
    - force_resource
    - resource
    - resource_area_equals
    - resource_scale
    - resource_unit
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - carrier_con_min
    - carrier_con_max
    - carrier_con_equals
    allowed_costs:
    - om_con
  hvac_transmission:
    inheritance:
    - transmission
    essentials:
      parent: transmission
      carrier: electricity
      color: '#CF74B2'
      name: Transmission intra-zonal electricity
      carrier_in: electricity
      carrier_out: electricity
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_con
    - energy_eff
    - energy_eff_per_distance
    - energy_prod
    - force_asynchronous_prod_con
    - lifetime
    - one_way
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - net_import_share_min
    - net_import_share_max
    - net_import_share_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - energy_cap_per_distance
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_prod
    - purchase
    - purchase_per_distance
  hydrogen_to_power:
    inheritance:
    - storage
    essentials:
      parent: storage
      carrier: electricity
      color: '#9958D6'
      name: Hydrogen-to-power storage
      carrier_in: electricity
      carrier_out: electricity
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - charge_rate
    - energy_cap_per_storage_cap_min
    - energy_cap_per_storage_cap_max
    - energy_cap_per_storage_cap_equals
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_con
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_asynchronous_prod_con
    - lifetime
    - storage_cap_equals
    - storage_cap_max
    - storage_cap_min
    - storage_cap_per_unit
    - storage_initial
    - storage_loss
    - storage_time_max
    - storage_discharge_depth
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - storage_cap_min
    - storage_cap_max
    - storage_cap_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_prod
    - purchase
    - storage_cap
  import_electricity:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#FF009E'
      name: Electricity import
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  lost_load:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#030303'
      name: Lost load
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  pv_farm:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#FFE400'
      name: Photovoltaic power farm-scale
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  pv_rooftop:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#FFB600'
      name: Photovoltaic power distributed
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  wind_offshore:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#275FB0'
      name: Off-shore wind power
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
  wind_onshore:
    inheritance:
    - supply
    essentials:
      parent: supply
      carrier_out: electricity
      color: '#27B09F'
      name: On-shore wind power
      carrier_in: resource
    constraints: {}
    required_constraints: []
    allowed_constraints:
    - energy_cap_equals
    - energy_cap_equals_systemwide
    - energy_cap_max
    - energy_cap_max_systemwide
    - energy_cap_min
    - energy_cap_min_use
    - energy_cap_per_unit
    - energy_cap_scale
    - energy_eff
    - energy_prod
    - energy_ramping
    - export_cap
    - export_carrier
    - force_resource
    - lifetime
    - resource
    - resource_area_equals
    - resource_area_max
    - resource_area_min
    - resource_area_per_energy_cap
    - resource_min_use
    - resource_scale
    - resource_unit
    - units_equals
    - units_equals_systemwide
    - units_max
    - units_max_systemwide
    - units_min
    allowed_group_constraints:
    - demand_share_min
    - demand_share_max
    - demand_share_equals
    - demand_share_per_timestep_min
    - demand_share_per_timestep_max
    - demand_share_per_timestep_equals
    - demand_share_per_timestep_decision
    - carrier_prod_share_min
    - carrier_prod_share_max
    - carrier_prod_share_equals
    - carrier_prod_share_per_timestep_min
    - carrier_prod_share_per_timestep_max
    - carrier_prod_share_per_timestep_equals
    - cost_max
    - cost_min
    - cost_equals
    - cost_var_max
    - cost_var_min
    - cost_var_equals
    - cost_investment_min
    - cost_investment_max
    - cost_investment_equals
    - energy_cap_share_min
    - energy_cap_share_max
    - energy_cap_share_equals
    - energy_cap_min
    - energy_cap_max
    - energy_cap_equals
    - resource_area_min
    - resource_area_max
    - resource_area_equals
    - carrier_prod_min
    - carrier_prod_max
    - carrier_prod_equals
    allowed_costs:
    - depreciation_rate
    - energy_cap
    - export
    - interest_rate
    - om_annual
    - om_annual_investment_fraction
    - om_con
    - om_prod
    - purchase
    - resource_area
tech_groups:
  conversion: []
  conversion_plus: []
  demand:
  - curtailment
  - demand_electricity
  - export_electricity
  storage:
  - battery
  - hydrogen_to_power
  supply:
  - ccgt
  - import_electricity
  - lost_load
  - pv_farm
  - pv_rooftop
  - wind_offshore
  - wind_onshore
  supply_plus: []
  transmission:
  - hvac_transmission
locations:
  NL11:
    coordinates:
      lat: 53.151875
      lon: 6.8779375
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL11
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL11
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL11
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL11
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL11
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL12:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 3.4401
              energy_cap_min: 0.6882
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 78.08059088268635
      NL13:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 51.40669619092387
  NL12:
    coordinates:
      lat: 53.130320000000005
      lon: 5.94212
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL12
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL12
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL12
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL12
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL12
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL11:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 3.4401
              energy_cap_min: 0.6882
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 78.08059088268635
      NL23:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 3.4401
              energy_cap_min: 0.6882
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 77.01279250805582
  NL13:
    coordinates:
      lat: 52.79966
      lon: 6.69054
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL13
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL13
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL13
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL13
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL11:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 51.40669619092387
      NL21:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 57.68592103711268
  NL21:
    coordinates:
      lat: 52.41295
      lon: 6.442433333333334
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL21
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL21
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL21
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL21
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL13:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 57.68592103711268
      NL22:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 74.3546324021931
      NL23:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 70.1574152900598
  NL22:
    coordinates:
      lat: 51.94764
      lon: 6.012
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL22
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL22
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL22
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL22
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL21:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 74.3546324021931
      NL41:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 91.64766157833785
  NL23:
    coordinates:
      lat: 52.60066666666666
      lon: 5.672633333333333
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL23
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL23
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL23
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL23
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL23
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL12:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 3.4401
              energy_cap_min: 0.6882
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 77.01279250805582
      NL21:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 15.3276
              energy_cap_min: 3.0655
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 70.1574152900598
      NL32:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 16.9811
              energy_cap_min: 3.3962
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 84.8212339513288
  NL31:
    coordinates:
      lat: 52.1608
      lon: 4.9873
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL31
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL31
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL31
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL31
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL32:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 34.96977818838089
      NL33:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 1.1886
              energy_cap_min: 0.2377
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 54.0333446482031
  NL32:
    coordinates:
      lat: 52.36538333333333
      lon: 4.748033333333333
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL32
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL32
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL32
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL32
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL32
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL23:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 16.9811
              energy_cap_min: 3.3962
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 84.8212339513288
      NL31:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 34.96977818838089
      NL33:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 62.54522684034896
  NL33:
    coordinates:
      lat: 51.95356666666667
      lon: 4.452383333333334
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL33
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL33
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL33
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL33
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL33
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL31:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 1.1886
              energy_cap_min: 0.2377
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 54.0333446482031
      NL32:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 62.54522684034896
      NL41:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 79.05131602490407
  NL34:
    coordinates:
      lat: 51.4261
      lon: 4.2185
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL34
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL34
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL34
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_offshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 30
          resource: file=windoff_nuts2.csv:NL34
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1060792482526339
            energy_cap: 1139.8826
            om_annual_investment_fraction: 2.3741
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL34
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL41:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 81.91534737849409
  NL41:
    coordinates:
      lat: 51.573025
      lon: 5.135325
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL41
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL41
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL41
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL41
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL22:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 91.64766157833785
      NL33:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 79.05131602490407
      NL34:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 11.8867
              energy_cap_min: 2.3773
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 81.91534737849409
      NL42:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 23.7734
              energy_cap_min: 4.7547
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 88.7810048549968
  NL42:
    coordinates:
      lat: 51.1524
      lon: 5.9052
    techs:
      battery:
        constraints:
          energy_cap_per_storage_cap_max: 0.25
          energy_con: true
          energy_eff: 0.98
          energy_prod: true
          lifetime: 22.5
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.11326683573865819
            om_annual_investment_fraction: 0.57
            om_prod: 0.002
            storage_cap: 885
      ccgt:
        constraints:
          energy_eff: 0.57
          energy_prod: true
          energy_ramping: 0.8
          lifetime: 25
          resource: inf
          resource_unit: energy
        costs:
          monetary:
            depreciation_rate: 0.11016807219002081
            energy_cap: 904.7795
            om_annual_investment_fraction: 3.3392
            om_con: 0.1
            om_prod: 0.00455
      curtailment:
        constraints:
          energy_con: true
          force_resource: false
          resource: -100000.0
          resource_unit: energy
        costs:
          monetary:
            om_con: 0
      demand_electricity:
        constraints:
          energy_con: true
          force_resource: true
          resource: file=electricity_demand_nuts2.csv:NL42
          resource_unit: energy
      hydrogen_to_power:
        constraints:
          energy_con: true
          energy_eff: 0.542
          energy_prod: true
          lifetime: 15
          storage_cap_max: inf
          storage_initial: 0
        costs:
          monetary:
            depreciation_rate: 0.13147377688737216
            energy_cap: 3069.866
            om_annual_investment_fraction: 4
            om_prod: 0
            storage_cap: 22.982
      lost_load:
        constraints:
          energy_prod: true
          force_resource: false
          lifetime: 20
          resource: 100000.0
          resource_unit: energy
        costs:
          monetary:
            om_prod: 10000.0
      pv_farm:
        constraints:
          energy_cap_max: 2500
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL42
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 676.5703
            om_annual_investment_fraction: 1.7275
            om_prod: 0
      pv_rooftop:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 37.5
          resource: file=pv_nuts2.csv:NL42
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.1028848670767948
            energy_cap: 880.0251
            om_annual_investment_fraction: 1.2567
            om_prod: 0
      wind_onshore:
        constraints:
          energy_cap_max: 5000
          energy_prod: true
          force_resource: true
          lifetime: 28.5
          resource: file=wind_nuts2.csv:NL42
          resource_unit: energy_per_cap
        costs:
          monetary:
            depreciation_rate: 0.10707971333223756
            energy_cap: 1139.8826
            om_annual_investment_fraction: 1.2347
            om_prod: 0
    links:
      NL41:
        techs:
          hvac_transmission:
            constraints:
              energy_cap_max: 23.7734
              energy_cap_min: 4.7547
              energy_con: true
              energy_eff: 0.96
              energy_prod: true
              lifetime: 60
            costs:
              monetary:
                depreciation_rate: 0.10032950922550408
                energy_cap: 175
                om_annual_investment_fraction: 0.015
            distance: 88.7810048549968
run:
  backend: pyomo
  bigM: 1000000.0
  cyclic_storage: true
  ensure_feasibility: true
  mode: plan
  objective_options:
    cost_class:
      monetary: 1
    sense: minimize
  objective: minmax_cost_optimization
  operation:
    window:
    horizon:
    use_cap_results: false
  spores_options:
    spores_number: 3
    slack: 0.1
    score_cost_class: spores_score
    objective_cost_class:
      spores_score: 1
      monetary: 0
    slack_cost_group:
    save_per_spore: false
    save_per_spore_path:
    skip_cost_op: false
  relax_constraint:
    demand_share_per_timestep_decision_main_constraint: 0
  save_logs:
  solver_io:
  solver_options:
  solver: cbc
  zero_threshold: 1e-10
model:
  calliope_version: 0.6.10
  group_share: {}
  name: Dutch power system at NUTS2 resolution
  random_seed:
  reserve_margin: {}
  subset_time:
  - '2019-01-01'
  - '2019-01-31'
  time:
    function: resample
    function_options:
      resolution: 3H
  timeseries_data_path: model_files/timeseries_data
  timeseries_data:
  timeseries_dateformat: '%Y-%m-%d %H:%M:%S'
  file_allowed:
  - clustering_func
  - energy_con
  - energy_eff
  - energy_prod
  - energy_ramping
  - export
  - force_resource
  - om_con
  - om_prod
  - parasitic_eff
  - resource
  - resource_eff
  - storage_loss
  - carrier_ratios
group_constraints: {}
sets:
  resources:
  - resource
  - electricity
  carriers:
  - electricity
  carrier_tiers:
  - out
  - in
  costs:
  - monetary
  locs:
  - NL12
  - NL22
  - NL23
  - NL31
  - NL41
  - NL42
  - NL11
  - NL32
  - NL21
  - NL33
  - NL13
  - NL34
  techs_non_transmission:
  - pv_rooftop
  - demand_electricity
  - wind_offshore
  - ccgt
  - export_electricity
  - pv_farm
  - battery
  - import_electricity
  - curtailment
  - hydrogen_to_power
  - lost_load
  - wind_onshore
  techs_demand:
  - demand_electricity
  - export_electricity
  - curtailment
  techs_supply:
  - pv_rooftop
  - wind_offshore
  - ccgt
  - pv_farm
  - import_electricity
  - lost_load
  - wind_onshore
  techs_supply_plus: []
  techs_conversion: []
  techs_conversion_plus: []
  techs_storage:
  - hydrogen_to_power
  - battery
  techs_transmission_names:
  - hvac_transmission
  techs_transmission:
  - hvac_transmission:NL13
  - hvac_transmission:NL42
  - hvac_transmission:NL32
  - hvac_transmission:NL22
  - hvac_transmission:NL31
  - hvac_transmission:NL12
  - hvac_transmission:NL34
  - hvac_transmission:NL23
  - hvac_transmission:NL33
  - hvac_transmission:NL21
  - hvac_transmission:NL41
  - hvac_transmission:NL11
  techs:
  - pv_rooftop
  - demand_electricity
  - wind_offshore
  - ccgt
  - export_electricity
  - pv_farm
  - battery
  - import_electricity
  - hvac_transmission
  - curtailment
  - hydrogen_to_power
  - lost_load
  - wind_onshore
  coordinates:
  - lat
  - lon
  timesteps:
  - '2019-01-01 00:00:00'
  - '2019-01-01 01:00:00'
  - '2019-01-01 02:00:00'
  - '2019-01-01 03:00:00'
  - '2019-01-01 04:00:00'
  - '2019-01-01 05:00:00'
  - '2019-01-01 06:00:00'
  - '2019-01-01 07:00:00'
  - '2019-01-01 08:00:00'
  - '2019-01-01 09:00:00'
  - '2019-01-01 10:00:00'
  - '2019-01-01 11:00:00'
  - '2019-01-01 12:00:00'
  - '2019-01-01 13:00:00'
  - '2019-01-01 14:00:00'
  - '2019-01-01 15:00:00'
  - '2019-01-01 16:00:00'
  - '2019-01-01 17:00:00'
  - '2019-01-01 18:00:00'
  - '2019-01-01 19:00:00'
  - '2019-01-01 20:00:00'
  - '2019-01-01 21:00:00'
  - '2019-01-01 22:00:00'
  - '2019-01-01 23:00:00'
  - '2019-01-02 00:00:00'
  - '2019-01-02 01:00:00'
  - '2019-01-02 02:00:00'
  - '2019-01-02 03:00:00'
  - '2019-01-02 04:00:00'
  - '2019-01-02 05:00:00'
  - '2019-01-02 06:00:00'
  - '2019-01-02 07:00:00'
  - '2019-01-02 08:00:00'
  - '2019-01-02 09:00:00'
  - '2019-01-02 10:00:00'
  - '2019-01-02 11:00:00'
  - '2019-01-02 12:00:00'
  - '2019-01-02 13:00:00'
  - '2019-01-02 14:00:00'
  - '2019-01-02 15:00:00'
  - '2019-01-02 16:00:00'
  - '2019-01-02 17:00:00'
  - '2019-01-02 18:00:00'
  - '2019-01-02 19:00:00'
  - '2019-01-02 20:00:00'
  - '2019-01-02 21:00:00'
  - '2019-01-02 22:00:00'
  - '2019-01-02 23:00:00'
  - '2019-01-03 00:00:00'
  - '2019-01-03 01:00:00'
  - '2019-01-03 02:00:00'
  - '2019-01-03 03:00:00'
  - '2019-01-03 04:00:00'
  - '2019-01-03 05:00:00'
  - '2019-01-03 06:00:00'
  - '2019-01-03 07:00:00'
  - '2019-01-03 08:00:00'
  - '2019-01-03 09:00:00'
  - '2019-01-03 10:00:00'
  - '2019-01-03 11:00:00'
  - '2019-01-03 12:00:00'
  - '2019-01-03 13:00:00'
  - '2019-01-03 14:00:00'
  - '2019-01-03 15:00:00'
  - '2019-01-03 16:00:00'
  - '2019-01-03 17:00:00'
  - '2019-01-03 18:00:00'
  - '2019-01-03 19:00:00'
  - '2019-01-03 20:00:00'
  - '2019-01-03 21:00:00'
  - '2019-01-03 22:00:00'
  - '2019-01-03 23:00:00'
  - '2019-01-04 00:00:00'
  - '2019-01-04 01:00:00'
  - '2019-01-04 02:00:00'
  - '2019-01-04 03:00:00'
  - '2019-01-04 04:00:00'
  - '2019-01-04 05:00:00'
  - '2019-01-04 06:00:00'
  - '2019-01-04 07:00:00'
  - '2019-01-04 08:00:00'
  - '2019-01-04 09:00:00'
  - '2019-01-04 10:00:00'
  - '2019-01-04 11:00:00'
  - '2019-01-04 12:00:00'
  - '2019-01-04 13:00:00'
  - '2019-01-04 14:00:00'
  - '2019-01-04 15:00:00'
  - '2019-01-04 16:00:00'
  - '2019-01-04 17:00:00'
  - '2019-01-04 18:00:00'
  - '2019-01-04 19:00:00'
  - '2019-01-04 20:00:00'
  - '2019-01-04 21:00:00'
  - '2019-01-04 22:00:00'
  - '2019-01-04 23:00:00'
  - '2019-01-05 00:00:00'
  - '2019-01-05 01:00:00'
  - '2019-01-05 02:00:00'
  - '2019-01-05 03:00:00'
  - '2019-01-05 04:00:00'
  - '2019-01-05 05:00:00'
  - '2019-01-05 06:00:00'
  - '2019-01-05 07:00:00'
  - '2019-01-05 08:00:00'
  - '2019-01-05 09:00:00'
  - '2019-01-05 10:00:00'
  - '2019-01-05 11:00:00'
  - '2019-01-05 12:00:00'
  - '2019-01-05 13:00:00'
  - '2019-01-05 14:00:00'
  - '2019-01-05 15:00:00'
  - '2019-01-05 16:00:00'
  - '2019-01-05 17:00:00'
  - '2019-01-05 18:00:00'
  - '2019-01-05 19:00:00'
  - '2019-01-05 20:00:00'
  - '2019-01-05 21:00:00'
  - '2019-01-05 22:00:00'
  - '2019-01-05 23:00:00'
  - '2019-01-06 00:00:00'
  - '2019-01-06 01:00:00'
  - '2019-01-06 02:00:00'
  - '2019-01-06 03:00:00'
  - '2019-01-06 04:00:00'
  - '2019-01-06 05:00:00'
  - '2019-01-06 06:00:00'
  - '2019-01-06 07:00:00'
  - '2019-01-06 08:00:00'
  - '2019-01-06 09:00:00'
  - '2019-01-06 10:00:00'
  - '2019-01-06 11:00:00'
  - '2019-01-06 12:00:00'
  - '2019-01-06 13:00:00'
  - '2019-01-06 14:00:00'
  - '2019-01-06 15:00:00'
  - '2019-01-06 16:00:00'
  - '2019-01-06 17:00:00'
  - '2019-01-06 18:00:00'
  - '2019-01-06 19:00:00'
  - '2019-01-06 20:00:00'
  - '2019-01-06 21:00:00'
  - '2019-01-06 22:00:00'
  - '2019-01-06 23:00:00'
  - '2019-01-07 00:00:00'
  - '2019-01-07 01:00:00'
  - '2019-01-07 02:00:00'
  - '2019-01-07 03:00:00'
  - '2019-01-07 04:00:00'
  - '2019-01-07 05:00:00'
  - '2019-01-07 06:00:00'
  - '2019-01-07 07:00:00'
  - '2019-01-07 08:00:00'
  - '2019-01-07 09:00:00'
  - '2019-01-07 10:00:00'
  - '2019-01-07 11:00:00'
  - '2019-01-07 12:00:00'
  - '2019-01-07 13:00:00'
  - '2019-01-07 14:00:00'
  - '2019-01-07 15:00:00'
  - '2019-01-07 16:00:00'
  - '2019-01-07 17:00:00'
  - '2019-01-07 18:00:00'
  - '2019-01-07 19:00:00'
  - '2019-01-07 20:00:00'
  - '2019-01-07 21:00:00'
  - '2019-01-07 22:00:00'
  - '2019-01-07 23:00:00'
  - '2019-01-08 00:00:00'
  - '2019-01-08 01:00:00'
  - '2019-01-08 02:00:00'
  - '2019-01-08 03:00:00'
  - '2019-01-08 04:00:00'
  - '2019-01-08 05:00:00'
  - '2019-01-08 06:00:00'
  - '2019-01-08 07:00:00'
  - '2019-01-08 08:00:00'
  - '2019-01-08 09:00:00'
  - '2019-01-08 10:00:00'
  - '2019-01-08 11:00:00'
  - '2019-01-08 12:00:00'
  - '2019-01-08 13:00:00'
  - '2019-01-08 14:00:00'
  - '2019-01-08 15:00:00'
  - '2019-01-08 16:00:00'
  - '2019-01-08 17:00:00'
  - '2019-01-08 18:00:00'
  - '2019-01-08 19:00:00'
  - '2019-01-08 20:00:00'
  - '2019-01-08 21:00:00'
  - '2019-01-08 22:00:00'
  - '2019-01-08 23:00:00'
  - '2019-01-09 00:00:00'
  - '2019-01-09 01:00:00'
  - '2019-01-09 02:00:00'
  - '2019-01-09 03:00:00'
  - '2019-01-09 04:00:00'
  - '2019-01-09 05:00:00'
  - '2019-01-09 06:00:00'
  - '2019-01-09 07:00:00'
  - '2019-01-09 08:00:00'
  - '2019-01-09 09:00:00'
  - '2019-01-09 10:00:00'
  - '2019-01-09 11:00:00'
  - '2019-01-09 12:00:00'
  - '2019-01-09 13:00:00'
  - '2019-01-09 14:00:00'
  - '2019-01-09 15:00:00'
  - '2019-01-09 16:00:00'
  - '2019-01-09 17:00:00'
  - '2019-01-09 18:00:00'
  - '2019-01-09 19:00:00'
  - '2019-01-09 20:00:00'
  - '2019-01-09 21:00:00'
  - '2019-01-09 22:00:00'
  - '2019-01-09 23:00:00'
  - '2019-01-10 00:00:00'
  - '2019-01-10 01:00:00'
  - '2019-01-10 02:00:00'
  - '2019-01-10 03:00:00'
  - '2019-01-10 04:00:00'
  - '2019-01-10 05:00:00'
  - '2019-01-10 06:00:00'
  - '2019-01-10 07:00:00'
  - '2019-01-10 08:00:00'
  - '2019-01-10 09:00:00'
  - '2019-01-10 10:00:00'
  - '2019-01-10 11:00:00'
  - '2019-01-10 12:00:00'
  - '2019-01-10 13:00:00'
  - '2019-01-10 14:00:00'
  - '2019-01-10 15:00:00'
  - '2019-01-10 16:00:00'
  - '2019-01-10 17:00:00'
  - '2019-01-10 18:00:00'
  - '2019-01-10 19:00:00'
  - '2019-01-10 20:00:00'
  - '2019-01-10 21:00:00'
  - '2019-01-10 22:00:00'
  - '2019-01-10 23:00:00'
  - '2019-01-11 00:00:00'
  - '2019-01-11 01:00:00'
  - '2019-01-11 02:00:00'
  - '2019-01-11 03:00:00'
  - '2019-01-11 04:00:00'
  - '2019-01-11 05:00:00'
  - '2019-01-11 06:00:00'
  - '2019-01-11 07:00:00'
  - '2019-01-11 08:00:00'
  - '2019-01-11 09:00:00'
  - '2019-01-11 10:00:00'
  - '2019-01-11 11:00:00'
  - '2019-01-11 12:00:00'
  - '2019-01-11 13:00:00'
  - '2019-01-11 14:00:00'
  - '2019-01-11 15:00:00'
  - '2019-01-11 16:00:00'
  - '2019-01-11 17:00:00'
  - '2019-01-11 18:00:00'
  - '2019-01-11 19:00:00'
  - '2019-01-11 20:00:00'
  - '2019-01-11 21:00:00'
  - '2019-01-11 22:00:00'
  - '2019-01-11 23:00:00'
  - '2019-01-12 00:00:00'
  - '2019-01-12 01:00:00'
  - '2019-01-12 02:00:00'
  - '2019-01-12 03:00:00'
  - '2019-01-12 04:00:00'
  - '2019-01-12 05:00:00'
  - '2019-01-12 06:00:00'
  - '2019-01-12 07:00:00'
  - '2019-01-12 08:00:00'
  - '2019-01-12 09:00:00'
  - '2019-01-12 10:00:00'
  - '2019-01-12 11:00:00'
  - '2019-01-12 12:00:00'
  - '2019-01-12 13:00:00'
  - '2019-01-12 14:00:00'
  - '2019-01-12 15:00:00'
  - '2019-01-12 16:00:00'
  - '2019-01-12 17:00:00'
  - '2019-01-12 18:00:00'
  - '2019-01-12 19:00:00'
  - '2019-01-12 20:00:00'
  - '2019-01-12 21:00:00'
  - '2019-01-12 22:00:00'
  - '2019-01-12 23:00:00'
  - '2019-01-13 00:00:00'
  - '2019-01-13 01:00:00'
  - '2019-01-13 02:00:00'
  - '2019-01-13 03:00:00'
  - '2019-01-13 04:00:00'
  - '2019-01-13 05:00:00'
  - '2019-01-13 06:00:00'
  - '2019-01-13 07:00:00'
  - '2019-01-13 08:00:00'
  - '2019-01-13 09:00:00'
  - '2019-01-13 10:00:00'
  - '2019-01-13 11:00:00'
  - '2019-01-13 12:00:00'
  - '2019-01-13 13:00:00'
  - '2019-01-13 14:00:00'
  - '2019-01-13 15:00:00'
  - '2019-01-13 16:00:00'
  - '2019-01-13 17:00:00'
  - '2019-01-13 18:00:00'
  - '2019-01-13 19:00:00'
  - '2019-01-13 20:00:00'
  - '2019-01-13 21:00:00'
  - '2019-01-13 22:00:00'
  - '2019-01-13 23:00:00'
  - '2019-01-14 00:00:00'
  - '2019-01-14 01:00:00'
  - '2019-01-14 02:00:00'
  - '2019-01-14 03:00:00'
  - '2019-01-14 04:00:00'
  - '2019-01-14 05:00:00'
  - '2019-01-14 06:00:00'
  - '2019-01-14 07:00:00'
  - '2019-01-14 08:00:00'
  - '2019-01-14 09:00:00'
  - '2019-01-14 10:00:00'
  - '2019-01-14 11:00:00'
  - '2019-01-14 12:00:00'
  - '2019-01-14 13:00:00'
  - '2019-01-14 14:00:00'
  - '2019-01-14 15:00:00'
  - '2019-01-14 16:00:00'
  - '2019-01-14 17:00:00'
  - '2019-01-14 18:00:00'
  - '2019-01-14 19:00:00'
  - '2019-01-14 20:00:00'
  - '2019-01-14 21:00:00'
  - '2019-01-14 22:00:00'
  - '2019-01-14 23:00:00'
  - '2019-01-15 00:00:00'
  - '2019-01-15 01:00:00'
  - '2019-01-15 02:00:00'
  - '2019-01-15 03:00:00'
  - '2019-01-15 04:00:00'
  - '2019-01-15 05:00:00'
  - '2019-01-15 06:00:00'
  - '2019-01-15 07:00:00'
  - '2019-01-15 08:00:00'
  - '2019-01-15 09:00:00'
  - '2019-01-15 10:00:00'
  - '2019-01-15 11:00:00'
  - '2019-01-15 12:00:00'
  - '2019-01-15 13:00:00'
  - '2019-01-15 14:00:00'
  - '2019-01-15 15:00:00'
  - '2019-01-15 16:00:00'
  - '2019-01-15 17:00:00'
  - '2019-01-15 18:00:00'
  - '2019-01-15 19:00:00'
  - '2019-01-15 20:00:00'
  - '2019-01-15 21:00:00'
  - '2019-01-15 22:00:00'
  - '2019-01-15 23:00:00'
  - '2019-01-16 00:00:00'
  - '2019-01-16 01:00:00'
  - '2019-01-16 02:00:00'
  - '2019-01-16 03:00:00'
  - '2019-01-16 04:00:00'
  - '2019-01-16 05:00:00'
  - '2019-01-16 06:00:00'
  - '2019-01-16 07:00:00'
  - '2019-01-16 08:00:00'
  - '2019-01-16 09:00:00'
  - '2019-01-16 10:00:00'
  - '2019-01-16 11:00:00'
  - '2019-01-16 12:00:00'
  - '2019-01-16 13:00:00'
  - '2019-01-16 14:00:00'
  - '2019-01-16 15:00:00'
  - '2019-01-16 16:00:00'
  - '2019-01-16 17:00:00'
  - '2019-01-16 18:00:00'
  - '2019-01-16 19:00:00'
  - '2019-01-16 20:00:00'
  - '2019-01-16 21:00:00'
  - '2019-01-16 22:00:00'
  - '2019-01-16 23:00:00'
  - '2019-01-17 00:00:00'
  - '2019-01-17 01:00:00'
  - '2019-01-17 02:00:00'
  - '2019-01-17 03:00:00'
  - '2019-01-17 04:00:00'
  - '2019-01-17 05:00:00'
  - '2019-01-17 06:00:00'
  - '2019-01-17 07:00:00'
  - '2019-01-17 08:00:00'
  - '2019-01-17 09:00:00'
  - '2019-01-17 10:00:00'
  - '2019-01-17 11:00:00'
  - '2019-01-17 12:00:00'
  - '2019-01-17 13:00:00'
  - '2019-01-17 14:00:00'
  - '2019-01-17 15:00:00'
  - '2019-01-17 16:00:00'
  - '2019-01-17 17:00:00'
  - '2019-01-17 18:00:00'
  - '2019-01-17 19:00:00'
  - '2019-01-17 20:00:00'
  - '2019-01-17 21:00:00'
  - '2019-01-17 22:00:00'
  - '2019-01-17 23:00:00'
  - '2019-01-18 00:00:00'
  - '2019-01-18 01:00:00'
  - '2019-01-18 02:00:00'
  - '2019-01-18 03:00:00'
  - '2019-01-18 04:00:00'
  - '2019-01-18 05:00:00'
  - '2019-01-18 06:00:00'
  - '2019-01-18 07:00:00'
  - '2019-01-18 08:00:00'
  - '2019-01-18 09:00:00'
  - '2019-01-18 10:00:00'
  - '2019-01-18 11:00:00'
  - '2019-01-18 12:00:00'
  - '2019-01-18 13:00:00'
  - '2019-01-18 14:00:00'
  - '2019-01-18 15:00:00'
  - '2019-01-18 16:00:00'
  - '2019-01-18 17:00:00'
  - '2019-01-18 18:00:00'
  - '2019-01-18 19:00:00'
  - '2019-01-18 20:00:00'
  - '2019-01-18 21:00:00'
  - '2019-01-18 22:00:00'
  - '2019-01-18 23:00:00'
  - '2019-01-19 00:00:00'
  - '2019-01-19 01:00:00'
  - '2019-01-19 02:00:00'
  - '2019-01-19 03:00:00'
  - '2019-01-19 04:00:00'
  - '2019-01-19 05:00:00'
  - '2019-01-19 06:00:00'
  - '2019-01-19 07:00:00'
  - '2019-01-19 08:00:00'
  - '2019-01-19 09:00:00'
  - '2019-01-19 10:00:00'
  - '2019-01-19 11:00:00'
  - '2019-01-19 12:00:00'
  - '2019-01-19 13:00:00'
  - '2019-01-19 14:00:00'
  - '2019-01-19 15:00:00'
  - '2019-01-19 16:00:00'
  - '2019-01-19 17:00:00'
  - '2019-01-19 18:00:00'
  - '2019-01-19 19:00:00'
  - '2019-01-19 20:00:00'
  - '2019-01-19 21:00:00'
  - '2019-01-19 22:00:00'
  - '2019-01-19 23:00:00'
  - '2019-01-20 00:00:00'
  - '2019-01-20 01:00:00'
  - '2019-01-20 02:00:00'
  - '2019-01-20 03:00:00'
  - '2019-01-20 04:00:00'
  - '2019-01-20 05:00:00'
  - '2019-01-20 06:00:00'
  - '2019-01-20 07:00:00'
  - '2019-01-20 08:00:00'
  - '2019-01-20 09:00:00'
  - '2019-01-20 10:00:00'
  - '2019-01-20 11:00:00'
  - '2019-01-20 12:00:00'
  - '2019-01-20 13:00:00'
  - '2019-01-20 14:00:00'
  - '2019-01-20 15:00:00'
  - '2019-01-20 16:00:00'
  - '2019-01-20 17:00:00'
  - '2019-01-20 18:00:00'
  - '2019-01-20 19:00:00'
  - '2019-01-20 20:00:00'
  - '2019-01-20 21:00:00'
  - '2019-01-20 22:00:00'
  - '2019-01-20 23:00:00'
  - '2019-01-21 00:00:00'
  - '2019-01-21 01:00:00'
  - '2019-01-21 02:00:00'
  - '2019-01-21 03:00:00'
  - '2019-01-21 04:00:00'
  - '2019-01-21 05:00:00'
  - '2019-01-21 06:00:00'
  - '2019-01-21 07:00:00'
  - '2019-01-21 08:00:00'
  - '2019-01-21 09:00:00'
  - '2019-01-21 10:00:00'
  - '2019-01-21 11:00:00'
  - '2019-01-21 12:00:00'
  - '2019-01-21 13:00:00'
  - '2019-01-21 14:00:00'
  - '2019-01-21 15:00:00'
  - '2019-01-21 16:00:00'
  - '2019-01-21 17:00:00'
  - '2019-01-21 18:00:00'
  - '2019-01-21 19:00:00'
  - '2019-01-21 20:00:00'
  - '2019-01-21 21:00:00'
  - '2019-01-21 22:00:00'
  - '2019-01-21 23:00:00'
  - '2019-01-22 00:00:00'
  - '2019-01-22 01:00:00'
  - '2019-01-22 02:00:00'
  - '2019-01-22 03:00:00'
  - '2019-01-22 04:00:00'
  - '2019-01-22 05:00:00'
  - '2019-01-22 06:00:00'
  - '2019-01-22 07:00:00'
  - '2019-01-22 08:00:00'
  - '2019-01-22 09:00:00'
  - '2019-01-22 10:00:00'
  - '2019-01-22 11:00:00'
  - '2019-01-22 12:00:00'
  - '2019-01-22 13:00:00'
  - '2019-01-22 14:00:00'
  - '2019-01-22 15:00:00'
  - '2019-01-22 16:00:00'
  - '2019-01-22 17:00:00'
  - '2019-01-22 18:00:00'
  - '2019-01-22 19:00:00'
  - '2019-01-22 20:00:00'
  - '2019-01-22 21:00:00'
  - '2019-01-22 22:00:00'
  - '2019-01-22 23:00:00'
  - '2019-01-23 00:00:00'
  - '2019-01-23 01:00:00'
  - '2019-01-23 02:00:00'
  - '2019-01-23 03:00:00'
  - '2019-01-23 04:00:00'
  - '2019-01-23 05:00:00'
  - '2019-01-23 06:00:00'
  - '2019-01-23 07:00:00'
  - '2019-01-23 08:00:00'
  - '2019-01-23 09:00:00'
  - '2019-01-23 10:00:00'
  - '2019-01-23 11:00:00'
  - '2019-01-23 12:00:00'
  - '2019-01-23 13:00:00'
  - '2019-01-23 14:00:00'
  - '2019-01-23 15:00:00'
  - '2019-01-23 16:00:00'
  - '2019-01-23 17:00:00'
  - '2019-01-23 18:00:00'
  - '2019-01-23 19:00:00'
  - '2019-01-23 20:00:00'
  - '2019-01-23 21:00:00'
  - '2019-01-23 22:00:00'
  - '2019-01-23 23:00:00'
  - '2019-01-24 00:00:00'
  - '2019-01-24 01:00:00'
  - '2019-01-24 02:00:00'
  - '2019-01-24 03:00:00'
  - '2019-01-24 04:00:00'
  - '2019-01-24 05:00:00'
  - '2019-01-24 06:00:00'
  - '2019-01-24 07:00:00'
  - '2019-01-24 08:00:00'
  - '2019-01-24 09:00:00'
  - '2019-01-24 10:00:00'
  - '2019-01-24 11:00:00'
  - '2019-01-24 12:00:00'
  - '2019-01-24 13:00:00'
  - '2019-01-24 14:00:00'
  - '2019-01-24 15:00:00'
  - '2019-01-24 16:00:00'
  - '2019-01-24 17:00:00'
  - '2019-01-24 18:00:00'
  - '2019-01-24 19:00:00'
  - '2019-01-24 20:00:00'
  - '2019-01-24 21:00:00'
  - '2019-01-24 22:00:00'
  - '2019-01-24 23:00:00'
  - '2019-01-25 00:00:00'
  - '2019-01-25 01:00:00'
  - '2019-01-25 02:00:00'
  - '2019-01-25 03:00:00'
  - '2019-01-25 04:00:00'
  - '2019-01-25 05:00:00'
  - '2019-01-25 06:00:00'
  - '2019-01-25 07:00:00'
  - '2019-01-25 08:00:00'
  - '2019-01-25 09:00:00'
  - '2019-01-25 10:00:00'
  - '2019-01-25 11:00:00'
  - '2019-01-25 12:00:00'
  - '2019-01-25 13:00:00'
  - '2019-01-25 14:00:00'
  - '2019-01-25 15:00:00'
  - '2019-01-25 16:00:00'
  - '2019-01-25 17:00:00'
  - '2019-01-25 18:00:00'
  - '2019-01-25 19:00:00'
  - '2019-01-25 20:00:00'
  - '2019-01-25 21:00:00'
  - '2019-01-25 22:00:00'
  - '2019-01-25 23:00:00'
  - '2019-01-26 00:00:00'
  - '2019-01-26 01:00:00'
  - '2019-01-26 02:00:00'
  - '2019-01-26 03:00:00'
  - '2019-01-26 04:00:00'
  - '2019-01-26 05:00:00'
  - '2019-01-26 06:00:00'
  - '2019-01-26 07:00:00'
  - '2019-01-26 08:00:00'
  - '2019-01-26 09:00:00'
  - '2019-01-26 10:00:00'
  - '2019-01-26 11:00:00'
  - '2019-01-26 12:00:00'
  - '2019-01-26 13:00:00'
  - '2019-01-26 14:00:00'
  - '2019-01-26 15:00:00'
  - '2019-01-26 16:00:00'
  - '2019-01-26 17:00:00'
  - '2019-01-26 18:00:00'
  - '2019-01-26 19:00:00'
  - '2019-01-26 20:00:00'
  - '2019-01-26 21:00:00'
  - '2019-01-26 22:00:00'
  - '2019-01-26 23:00:00'
  - '2019-01-27 00:00:00'
  - '2019-01-27 01:00:00'
  - '2019-01-27 02:00:00'
  - '2019-01-27 03:00:00'
  - '2019-01-27 04:00:00'
  - '2019-01-27 05:00:00'
  - '2019-01-27 06:00:00'
  - '2019-01-27 07:00:00'
  - '2019-01-27 08:00:00'
  - '2019-01-27 09:00:00'
  - '2019-01-27 10:00:00'
  - '2019-01-27 11:00:00'
  - '2019-01-27 12:00:00'
  - '2019-01-27 13:00:00'
  - '2019-01-27 14:00:00'
  - '2019-01-27 15:00:00'
  - '2019-01-27 16:00:00'
  - '2019-01-27 17:00:00'
  - '2019-01-27 18:00:00'
  - '2019-01-27 19:00:00'
  - '2019-01-27 20:00:00'
  - '2019-01-27 21:00:00'
  - '2019-01-27 22:00:00'
  - '2019-01-27 23:00:00'
  - '2019-01-28 00:00:00'
  - '2019-01-28 01:00:00'
  - '2019-01-28 02:00:00'
  - '2019-01-28 03:00:00'
  - '2019-01-28 04:00:00'
  - '2019-01-28 05:00:00'
  - '2019-01-28 06:00:00'
  - '2019-01-28 07:00:00'
  - '2019-01-28 08:00:00'
  - '2019-01-28 09:00:00'
  - '2019-01-28 10:00:00'
  - '2019-01-28 11:00:00'
  - '2019-01-28 12:00:00'
  - '2019-01-28 13:00:00'
  - '2019-01-28 14:00:00'
  - '2019-01-28 15:00:00'
  - '2019-01-28 16:00:00'
  - '2019-01-28 17:00:00'
  - '2019-01-28 18:00:00'
  - '2019-01-28 19:00:00'
  - '2019-01-28 20:00:00'
  - '2019-01-28 21:00:00'
  - '2019-01-28 22:00:00'
  - '2019-01-28 23:00:00'
  - '2019-01-29 00:00:00'
  - '2019-01-29 01:00:00'
  - '2019-01-29 02:00:00'
  - '2019-01-29 03:00:00'
  - '2019-01-29 04:00:00'
  - '2019-01-29 05:00:00'
  - '2019-01-29 06:00:00'
  - '2019-01-29 07:00:00'
  - '2019-01-29 08:00:00'
  - '2019-01-29 09:00:00'
  - '2019-01-29 10:00:00'
  - '2019-01-29 11:00:00'
  - '2019-01-29 12:00:00'
  - '2019-01-29 13:00:00'
  - '2019-01-29 14:00:00'
  - '2019-01-29 15:00:00'
  - '2019-01-29 16:00:00'
  - '2019-01-29 17:00:00'
  - '2019-01-29 18:00:00'
  - '2019-01-29 19:00:00'
  - '2019-01-29 20:00:00'
  - '2019-01-29 21:00:00'
  - '2019-01-29 22:00:00'
  - '2019-01-29 23:00:00'
  - '2019-01-30 00:00:00'
  - '2019-01-30 01:00:00'
  - '2019-01-30 02:00:00'
  - '2019-01-30 03:00:00'
  - '2019-01-30 04:00:00'
  - '2019-01-30 05:00:00'
  - '2019-01-30 06:00:00'
  - '2019-01-30 07:00:00'
  - '2019-01-30 08:00:00'
  - '2019-01-30 09:00:00'
  - '2019-01-30 10:00:00'
  - '2019-01-30 11:00:00'
  - '2019-01-30 12:00:00'
  - '2019-01-30 13:00:00'
  - '2019-01-30 14:00:00'
  - '2019-01-30 15:00:00'
  - '2019-01-30 16:00:00'
  - '2019-01-30 17:00:00'
  - '2019-01-30 18:00:00'
  - '2019-01-30 19:00:00'
  - '2019-01-30 20:00:00'
  - '2019-01-30 21:00:00'
  - '2019-01-30 22:00:00'
  - '2019-01-30 23:00:00'
  - '2019-01-31 00:00:00'
  - '2019-01-31 01:00:00'
  - '2019-01-31 02:00:00'
  - '2019-01-31 03:00:00'
  - '2019-01-31 04:00:00'
  - '2019-01-31 05:00:00'
  - '2019-01-31 06:00:00'
  - '2019-01-31 07:00:00'
  - '2019-01-31 08:00:00'
  - '2019-01-31 09:00:00'
  - '2019-01-31 10:00:00'
  - '2019-01-31 11:00:00'
  - '2019-01-31 12:00:00'
  - '2019-01-31 13:00:00'
  - '2019-01-31 14:00:00'
  - '2019-01-31 15:00:00'
  - '2019-01-31 16:00:00'
  - '2019-01-31 17:00:00'
  - '2019-01-31 18:00:00'
  - '2019-01-31 19:00:00'
  - '2019-01-31 20:00:00'
  - '2019-01-31 21:00:00'
  - '2019-01-31 22:00:00'
  - '2019-01-31 23:00:00'
  techlists: []
  loc_carriers:
  - NL11::electricity
  - NL22::electricity
  - NL33::electricity
  - NL23::electricity
  - NL42::electricity
  - NL12::electricity
  - NL21::electricity
  - NL34::electricity
  - NL13::electricity
  - NL31::electricity
  - NL32::electricity
  - NL41::electricity
  loc_tech_carriers_con:
  - NL41::hvac_transmission:NL42::electricity
  - NL41::hvac_transmission:NL22::electricity
  - NL32::hvac_transmission:NL31::electricity
  - NL33::battery::electricity
  - NL21::battery::electricity
  - NL32::hvac_transmission:NL23::electricity
  - NL23::hvac_transmission:NL32::electricity
  - NL13::hvac_transmission:NL11::electricity
  - NL33::curtailment::electricity
  - NL11::curtailment::electricity
  - NL42::hydrogen_to_power::electricity
  - NL22::battery::electricity
  - NL34::demand_electricity::electricity
  - NL11::demand_electricity::electricity
  - NL21::curtailment::electricity
  - NL42::curtailment::electricity
  - NL34::hydrogen_to_power::electricity
  - NL31::hvac_transmission:NL33::electricity
  - NL34::battery::electricity
  - NL21::hydrogen_to_power::electricity
  - NL13::battery::electricity
  - NL32::demand_electricity::electricity
  - NL42::demand_electricity::electricity
  - NL21::hvac_transmission:NL23::electricity
  - NL41::hvac_transmission:NL33::electricity
  - NL33::hvac_transmission:NL31::electricity
  - NL11::hvac_transmission:NL13::electricity
  - NL32::battery::electricity
  - NL31::hydrogen_to_power::electricity
  - NL13::hydrogen_to_power::electricity
  - NL34::hvac_transmission:NL41::electricity
  - NL33::hydrogen_to_power::electricity
  - NL33::demand_electricity::electricity
  - NL42::hvac_transmission:NL41::electricity
  - NL32::hydrogen_to_power::electricity
  - NL11::hvac_transmission:NL12::electricity
  - NL12::hvac_transmission:NL11::electricity
  - NL33::hvac_transmission:NL32::electricity
  - NL41::hvac_transmission:NL34::electricity
  - NL21::demand_electricity::electricity
  - NL22::demand_electricity::electricity
  - NL21::hvac_transmission:NL22::electricity
  - NL23::curtailment::electricity
  - NL41::hydrogen_to_power::electricity
  - NL32::hvac_transmission:NL33::electricity
  - NL13::curtailment::electricity
  - NL23::hvac_transmission:NL21::electricity
  - NL12::demand_electricity::electricity
  - NL23::hydrogen_to_power::electricity
  - NL22::hvac_transmission:NL21::electricity
  - NL31::battery::electricity
  - NL22::hydrogen_to_power::electricity
  - NL31::curtailment::electricity
  - NL41::battery::electricity
  - NL23::battery::electricity
  - NL13::demand_electricity::electricity
  - NL12::battery::electricity
  - NL33::hvac_transmission:NL41::electricity
  - NL23::demand_electricity::electricity
  - NL34::curtailment::electricity
  - NL41::curtailment::electricity
  - NL31::demand_electricity::electricity
  - NL11::hydrogen_to_power::electricity
  - NL12::curtailment::electricity
  - NL31::hvac_transmission:NL32::electricity
  - NL12::hvac_transmission:NL23::electricity
  - NL22::hvac_transmission:NL41::electricity
  - NL21::hvac_transmission:NL13::electricity
  - NL32::curtailment::electricity
  - NL11::battery::electricity
  - NL23::hvac_transmission:NL12::electricity
  - NL12::hydrogen_to_power::electricity
  - NL13::hvac_transmission:NL21::electricity
  - NL42::battery::electricity
  - NL41::demand_electricity::electricity
  - NL22::curtailment::electricity
  loc_tech_carriers_conversion_all: []
  loc_tech_carriers_conversion_plus: []
  loc_tech_carriers_demand:
  - NL11::curtailment::electricity
  - NL34::demand_electricity::electricity
  - NL11::demand_electricity::electricity
  - NL21::curtailment::electricity
  - NL42::curtailment::electricity
  - NL32::demand_electricity::electricity
  - NL42::demand_electricity::electricity
  - NL33::demand_electricity::electricity
  - NL21::demand_electricity::electricity
  - NL33::curtailment::electricity
  - NL22::demand_electricity::electricity
  - NL23::curtailment::electricity
  - NL13::curtailment::electricity
  - NL12::demand_electricity::electricity
  - NL31::curtailment::electricity
  - NL13::demand_electricity::electricity
  - NL23::demand_electricity::electricity
  - NL34::curtailment::electricity
  - NL41::curtailment::electricity
  - NL31::demand_electricity::electricity
  - NL12::curtailment::electricity
  - NL32::curtailment::electricity
  - NL41::demand_electricity::electricity
  - NL22::curtailment::electricity
  loc_tech_carriers_export: []
  loc_tech_carriers_prod:
  - NL33::battery::electricity
  - NL22::pv_farm::electricity
  - NL42::hydrogen_to_power::electricity
  - NL22::battery::electricity
  - NL42::pv_rooftop::electricity
  - NL21::hydrogen_to_power::electricity
  - NL41::hvac_transmission:NL33::electricity
  - NL32::ccgt::electricity
  - NL13::pv_rooftop::electricity
  - NL22::hvac_transmission:NL21::electricity
  - NL21::wind_onshore::electricity
  - NL41::battery::electricity
  - NL23::battery::electricity
  - NL12::battery::electricity
  - NL32::wind_offshore::electricity
  - NL31::hvac_transmission:NL32::electricity
  - NL12::hvac_transmission:NL23::electricity
  - NL21::hvac_transmission:NL13::electricity
  - NL12::pv_farm::electricity
  - NL12::hydrogen_to_power::electricity
  - NL32::hydrogen_to_power::electricity
  - NL13::hvac_transmission:NL11::electricity
  - NL13::hvac_transmission:NL21::electricity
  - NL34::wind_offshore::electricity
  - NL41::hvac_transmission:NL42::electricity
  - NL32::hvac_transmission:NL31::electricity
  - NL34::pv_rooftop::electricity
  - NL21::battery::electricity
  - NL41::wind_onshore::electricity
  - NL12::ccgt::electricity
  - NL34::hydrogen_to_power::electricity
  - NL21::lost_load::electricity
  - NL32::battery::electricity
  - NL34::pv_farm::electricity
  - NL11::hvac_transmission:NL12::electricity
  - NL41::hvac_transmission:NL34::electricity
  - NL33::wind_offshore::electricity
  - NL21::hvac_transmission:NL22::electricity
  - NL41::hydrogen_to_power::electricity
  - NL13::lost_load::electricity
  - NL33::lost_load::electricity
  - NL42::ccgt::electricity
  - NL12::wind_onshore::electricity
  - NL21::pv_farm::electricity
  - NL41::ccgt::electricity
  - NL12::wind_offshore::electricity
  - NL11::wind_onshore::electricity
  - NL31::battery::electricity
  - NL11::pv_rooftop::electricity
  - NL23::lost_load::electricity
  - NL31::ccgt::electricity
  - NL41::pv_farm::electricity
  - NL42::wind_onshore::electricity
  - NL33::ccgt::electricity
  - NL31::wind_onshore::electricity
  - NL22::ccgt::electricity
  - NL33::hvac_transmission:NL31::electricity
  - NL11::hvac_transmission:NL13::electricity
  - NL22::pv_rooftop::electricity
  - NL11::wind_offshore::electricity
  - NL31::hydrogen_to_power::electricity
  - NL41::lost_load::electricity
  - NL23::pv_farm::electricity
  - NL11::lost_load::electricity
  - NL33::hydrogen_to_power::electricity
  - NL31::pv_rooftop::electricity
  - NL33::hvac_transmission:NL32::electricity
  - NL32::hvac_transmission:NL33::electricity
  - NL22::hydrogen_to_power::electricity
  - NL22::lost_load::electricity
  - NL32::lost_load::electricity
  - NL33::hvac_transmission:NL41::electricity
  - NL22::wind_onshore::electricity
  - NL41::pv_rooftop::electricity
  - NL33::wind_onshore::electricity
  - NL12::pv_rooftop::electricity
  - NL33::pv_rooftop::electricity
  - NL11::pv_farm::electricity
  - NL23::pv_rooftop::electricity
  - NL23::ccgt::electricity
  - NL11::battery::electricity
  - NL32::pv_farm::electricity
  - NL41::hvac_transmission:NL22::electricity
  - NL34::wind_onshore::electricity
  - NL32::hvac_transmission:NL23::electricity
  - NL23::hvac_transmission:NL32::electricity
  - NL21::ccgt::electricity
  - NL13::ccgt::electricity
  - NL31::hvac_transmission:NL33::electricity
  - NL34::battery::electricity
  - NL23::wind_offshore::electricity
  - NL34::lost_load::electricity
  - NL21::hvac_transmission:NL23::electricity
  - NL11::ccgt::electricity
  - NL31::pv_farm::electricity
  - NL13::hydrogen_to_power::electricity
  - NL34::hvac_transmission:NL41::electricity
  - NL42::hvac_transmission:NL41::electricity
  - NL42::lost_load::electricity
  - NL12::hvac_transmission:NL11::electricity
  - NL13::wind_onshore::electricity
  - NL34::ccgt::electricity
  - NL12::lost_load::electricity
  - NL13::battery::electricity
  - NL23::hydrogen_to_power::electricity
  - NL23::hvac_transmission:NL21::electricity
  - NL31::lost_load::electricity
  - NL21::pv_rooftop::electricity
  - NL13::pv_farm::electricity
  - NL23::wind_onshore::electricity
  - NL32::pv_rooftop::electricity
  - NL32::wind_onshore::electricity
  - NL11::hydrogen_to_power::electricity
  - NL33::pv_farm::electricity
  - NL42::pv_farm::electricity
  - NL22::hvac_transmission:NL41::electricity
  - NL23::hvac_transmission:NL12::electricity
  - NL42::battery::electricity
  loc_tech_carriers_supply_all:
  - NL32::pv_farm::electricity
  - NL34::wind_onshore::electricity
  - NL34::pv_rooftop::electricity
  - NL41::wind_onshore::electricity
  - NL22::pv_farm::electricity
  - NL21::ccgt::electricity
  - NL13::ccgt::electricity
  - NL31::ccgt::electricity
  - NL42::pv_rooftop::electricity
  - NL41::pv_farm::electricity
  - NL12::ccgt::electricity
  - NL23::wind_offshore::electricity
  - NL42::wind_onshore::electricity
  - NL33::ccgt::electricity
  - NL21::lost_load::electricity
  - NL34::lost_load::electricity
  - NL31::wind_onshore::electricity
  - NL22::ccgt::electricity
  - NL11::ccgt::electricity
  - NL31::pv_farm::electricity
  - NL22::pv_rooftop::electricity
  - NL11::wind_offshore::electricity
  - NL41::lost_load::electricity
  - NL23::pv_farm::electricity
  - NL11::lost_load::electricity
  - NL31::pv_rooftop::electricity
  - NL32::ccgt::electricity
  - NL34::pv_farm::electricity
  - NL42::lost_load::electricity
  - NL13::wind_onshore::electricity
  - NL33::wind_offshore::electricity
  - NL12::lost_load::electricity
  - NL13::pv_rooftop::electricity
  - NL21::pv_farm::electricity
  - NL34::ccgt::electricity
  - NL31::lost_load::electricity
  - NL21::pv_rooftop::electricity
  - NL21::wind_onshore::electricity
  - NL13::lost_load::electricity
  - NL23::wind_onshore::electricity
  - NL13::pv_farm::electricity
  - NL32::pv_rooftop::electricity
  - NL32::wind_onshore::electricity
  - NL22::lost_load::electricity
  - NL32::lost_load::electricity
  - NL32::wind_offshore::electricity
  - NL22::wind_onshore::electricity
  - NL41::pv_rooftop::electricity
  - NL33::wind_onshore::electricity
  - NL33::lost_load::electricity
  - NL42::ccgt::electricity
  - NL12::pv_rooftop::electricity
  - NL33::pv_farm::electricity
  - NL12::wind_onshore::electricity
  - NL42::pv_farm::electricity
  - NL41::ccgt::electricity
  - NL12::pv_farm::electricity
  - NL12::wind_offshore::electricity
  - NL23::pv_rooftop::electricity
  - NL23::ccgt::electricity
  - NL11::pv_farm::electricity
  - NL33::pv_rooftop::electricity
  - NL11::wind_onshore::electricity
  - NL11::pv_rooftop::electricity
  - NL23::lost_load::electricity
  - NL34::wind_offshore::electricity
  loc_tech_carriers_supply_conversion_all:
  - NL22::pv_farm::electricity
  - NL31::ccgt::electricity
  - NL42::pv_rooftop::electricity
  - NL41::pv_farm::electricity
  - NL42::wind_onshore::electricity
  - NL33::ccgt::electricity
  - NL31::wind_onshore::electricity
  - NL22::ccgt::electricity
  - NL22::pv_rooftop::electricity
  - NL11::wind_offshore::electricity
  - NL41::lost_load::electricity
  - NL23::pv_farm::electricity
  - NL11::lost_load::electricity
  - NL31::pv_rooftop::electricity
  - NL32::ccgt::electricity
  - NL13::pv_rooftop::electricity
  - NL21::wind_onshore::electricity
  - NL22::lost_load::electricity
  - NL32::lost_load::electricity
  - NL32::wind_offshore::electricity
  - NL22::wind_onshore::electricity
  - NL41::pv_rooftop::electricity
  - NL33::wind_onshore::electricity
  - NL12::pv_rooftop::electricity
  - NL33::pv_rooftop::electricity
  - NL12::pv_farm::electricity
  - NL11::pv_farm::electricity
  - NL23::pv_rooftop::electricity
  - NL23::ccgt::electricity
  - NL34::wind_offshore::electricity
  - NL32::pv_farm::electricity
  - NL34::wind_onshore::electricity
  - NL34::pv_rooftop::electricity
  - NL41::wind_onshore::electricity
  - NL21::ccgt::electricity
  - NL13::ccgt::electricity
  - NL12::ccgt::electricity
  - NL23::wind_offshore::electricity
  - NL21::lost_load::electricity
  - NL34::lost_load::electricity
  - NL11::ccgt::electricity
  - NL31::pv_farm::electricity
  - NL34::pv_farm::electricity
  - NL42::lost_load::electricity
  - NL13::wind_onshore::electricity
  - NL33::wind_offshore::electricity
  - NL12::lost_load::electricity
  - NL34::ccgt::electricity
  - NL31::lost_load::electricity
  - NL21::pv_rooftop::electricity
  - NL13::lost_load::electricity
  - NL23::wind_onshore::electricity
  - NL13::pv_farm::electricity
  - NL32::pv_rooftop::electricity
  - NL32::wind_onshore::electricity
  - NL33::lost_load::electricity
  - NL42::ccgt::electricity
  - NL33::pv_farm::electricity
  - NL42::pv_farm::electricity
  - NL21::pv_farm::electricity
  - NL12::wind_onshore::electricity
  - NL41::ccgt::electricity
  - NL12::wind_offshore::electricity
  - NL11::wind_onshore::electricity
  - NL11::pv_rooftop::electricity
  - NL23::lost_load::electricity
  loc_techs:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL34::battery
  - NL34::curtailment
  - NL42::hydrogen_to_power
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL41::hvac_transmission:NL42
  - NL23::demand_electricity
  - NL22::wind_onshore
  - NL21::hvac_transmission:NL13
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL41::hvac_transmission:NL34
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL33
  - NL23::curtailment
  - NL11::wind_offshore
  - NL32::hvac_transmission:NL23
  - NL33::hvac_transmission:NL41
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::demand_electricity
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL22::lost_load
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL21::demand_electricity
  - NL32::wind_offshore
  - NL42::demand_electricity
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL33::demand_electricity
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::demand_electricity
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hvac_transmission:NL12
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL34::demand_electricity
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL22::hvac_transmission:NL41
  - NL32::hvac_transmission:NL31
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL31::demand_electricity
  - NL33::wind_onshore
  - NL31::battery
  - NL12::lost_load
  - NL31::pv_rooftop
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL41::hvac_transmission:NL33
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_area: []
  loc_techs_asynchronous_prod_con: []
  loc_techs_conversion: []
  loc_techs_conversion_all: []
  loc_techs_conversion_plus: []
  loc_techs_cost:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL34::curtailment
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL21::hvac_transmission:NL22
  - NL42::hvac_transmission:NL41
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL41::hvac_transmission:NL42
  - NL21::hvac_transmission:NL13
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL41::hvac_transmission:NL34
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL11::wind_offshore
  - NL33::hvac_transmission:NL41
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL22::lost_load
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL33::hvac_transmission:NL31
  - NL23::hvac_transmission:NL12
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL22::hvac_transmission:NL41
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL42::wind_onshore
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL32::hvac_transmission:NL31
  - NL22::pv_rooftop
  - NL41::hvac_transmission:NL33
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL23::curtailment
  - NL34::pv_farm
  loc_techs_costs_export: []
  loc_techs_demand:
  - NL21::demand_electricity
  - NL42::demand_electricity
  - NL33::demand_electricity
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL12::demand_electricity
  - NL34::curtailment
  - NL21::curtailment
  - NL41::curtailment
  - NL42::curtailment
  - NL23::demand_electricity
  - NL34::demand_electricity
  - NL33::curtailment
  - NL22::curtailment
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL31::curtailment
  - NL32::curtailment
  - NL31::demand_electricity
  - NL32::demand_electricity
  - NL12::curtailment
  - NL11::curtailment
  - NL13::curtailment
  - NL23::curtailment
  loc_techs_export: []
  loc_techs_finite_resource:
  - NL41::pv_farm
  - NL31::wind_onshore
  - NL23::pv_rooftop
  - NL11::demand_electricity
  - NL34::curtailment
  - NL42::curtailment
  - NL23::demand_electricity
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL23::lost_load
  - NL13::lost_load
  - NL22::pv_farm
  - NL11::wind_offshore
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::demand_electricity
  - NL12::curtailment
  - NL22::lost_load
  - NL21::demand_electricity
  - NL42::demand_electricity
  - NL12::pv_rooftop
  - NL34::wind_onshore
  - NL33::demand_electricity
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL11::lost_load
  - NL21::wind_onshore
  - NL34::demand_electricity
  - NL33::pv_farm
  - NL42::lost_load
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL31::demand_electricity
  - NL12::lost_load
  - NL11::curtailment
  - NL23::pv_farm
  - NL31::pv_farm
  - NL23::curtailment
  - NL42::wind_onshore
  - NL34::pv_farm
  - NL23::wind_onshore
  - NL41::pv_rooftop
  - NL13::demand_electricity
  - NL41::curtailment
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL13::pv_farm
  - NL33::lost_load
  - NL32::wind_onshore
  - NL21::lost_load
  - NL23::wind_offshore
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL32::wind_offshore
  - NL12::demand_electricity
  - NL34::pv_rooftop
  - NL21::curtailment
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL34::lost_load
  - NL33::curtailment
  - NL22::curtailment
  - NL42::pv_farm
  - NL31::curtailment
  - NL32::curtailment
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::pv_rooftop
  - NL22::pv_rooftop
  - NL11::pv_rooftop
  loc_techs_finite_resource_demand:
  - NL21::demand_electricity
  - NL42::demand_electricity
  - NL33::demand_electricity
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL12::demand_electricity
  - NL34::curtailment
  - NL21::curtailment
  - NL41::curtailment
  - NL42::curtailment
  - NL23::demand_electricity
  - NL34::demand_electricity
  - NL33::curtailment
  - NL22::curtailment
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL31::curtailment
  - NL32::curtailment
  - NL31::demand_electricity
  - NL32::demand_electricity
  - NL12::curtailment
  - NL11::curtailment
  - NL13::curtailment
  - NL23::curtailment
  loc_techs_finite_resource_supply:
  - NL41::pv_farm
  - NL23::wind_onshore
  - NL32::wind_offshore
  - NL31::wind_onshore
  - NL12::pv_rooftop
  - NL41::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL23::pv_rooftop
  - NL31::pv_farm
  - NL34::pv_rooftop
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL12::wind_onshore
  - NL11::lost_load
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::wind_onshore
  - NL11::pv_farm
  - NL32::lost_load
  - NL32::pv_farm
  - NL33::pv_rooftop
  - NL23::lost_load
  - NL13::lost_load
  - NL32::pv_rooftop
  - NL21::wind_onshore
  - NL22::pv_farm
  - NL33::pv_farm
  - NL11::wind_offshore
  - NL34::lost_load
  - NL42::lost_load
  - NL13::pv_farm
  - NL42::pv_farm
  - NL33::lost_load
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL33::wind_offshore
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL41::lost_load
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL32::wind_onshore
  - NL22::pv_rooftop
  - NL21::lost_load
  - NL11::pv_rooftop
  - NL23::wind_offshore
  - NL23::pv_farm
  - NL42::pv_rooftop
  - NL42::wind_onshore
  - NL34::pv_farm
  - NL22::lost_load
  loc_techs_finite_resource_supply_plus: []
  loc_techs_in_2: []
  loc_techs_in_3: []
  loc_techs_investment_cost:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL21::hvac_transmission:NL22
  - NL42::hvac_transmission:NL41
  - NL41::ccgt
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL41::hvac_transmission:NL42
  - NL12::pv_farm
  - NL23::battery
  - NL21::hvac_transmission:NL13
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL31::hydrogen_to_power
  - NL41::hvac_transmission:NL34
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL32::hvac_transmission:NL33
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL23
  - NL11::wind_offshore
  - NL33::hvac_transmission:NL41
  - NL13::pv_farm
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::wind_onshore
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hvac_transmission:NL12
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::pv_farm
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL22::hvac_transmission:NL41
  - NL42::pv_farm
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL34::wind_offshore
  - NL11::ccgt
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL32::hvac_transmission:NL31
  - NL34::ccgt
  - NL41::hvac_transmission:NL33
  - NL22::pv_rooftop
  - NL22::hydrogen_to_power
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_milp: []
  loc_techs_non_conversion:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL34::battery
  - NL34::curtailment
  - NL42::hydrogen_to_power
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL41::hvac_transmission:NL42
  - NL23::demand_electricity
  - NL22::wind_onshore
  - NL21::hvac_transmission:NL13
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL41::hvac_transmission:NL34
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL11::wind_offshore
  - NL33::hvac_transmission:NL41
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::demand_electricity
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL22::lost_load
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL21::demand_electricity
  - NL32::wind_offshore
  - NL42::demand_electricity
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL33::demand_electricity
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::demand_electricity
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hvac_transmission:NL12
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL34::demand_electricity
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL22::hvac_transmission:NL41
  - NL32::hvac_transmission:NL31
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL42::wind_onshore
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL31::demand_electricity
  - NL33::wind_onshore
  - NL31::battery
  - NL12::lost_load
  - NL31::pv_rooftop
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL41::hvac_transmission:NL33
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL23::curtailment
  - NL34::pv_farm
  loc_techs_non_transmission:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL31::wind_onshore
  - NL23::pv_rooftop
  - NL11::demand_electricity
  - NL34::battery
  - NL34::curtailment
  - NL42::curtailment
  - NL23::demand_electricity
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL23::lost_load
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL12::ccgt
  - NL22::pv_farm
  - NL11::wind_offshore
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::demand_electricity
  - NL12::curtailment
  - NL22::lost_load
  - NL21::demand_electricity
  - NL42::demand_electricity
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL33::demand_electricity
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL11::lost_load
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL34::demand_electricity
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL42::lost_load
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL11::ccgt
  - NL31::demand_electricity
  - NL12::lost_load
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL13::battery
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL23::curtailment
  - NL34::pv_farm
  - NL23::wind_onshore
  - NL12::battery
  - NL41::pv_rooftop
  - NL32::battery
  - NL13::demand_electricity
  - NL42::hydrogen_to_power
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL31::ccgt
  - NL31::hydrogen_to_power
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL13::pv_farm
  - NL33::lost_load
  - NL42::battery
  - NL32::wind_onshore
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL33::battery
  - NL32::wind_offshore
  - NL12::demand_electricity
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL22::ccgt
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL34::lost_load
  - NL33::curtailment
  - NL22::curtailment
  - NL42::pv_farm
  - NL31::curtailment
  - NL32::curtailment
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL11::pv_rooftop
  loc_techs_om_cost:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL34::curtailment
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL23::curtailment
  - NL11::wind_offshore
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL22::lost_load
  - NL33::battery
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_om_cost_conversion: []
  loc_techs_om_cost_conversion_plus: []
  loc_techs_om_cost_supply:
  - NL41::pv_farm
  - NL23::wind_onshore
  - NL21::ccgt
  - NL32::wind_offshore
  - NL31::wind_onshore
  - NL12::pv_rooftop
  - NL41::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL23::pv_rooftop
  - NL13::ccgt
  - NL31::pv_farm
  - NL34::pv_rooftop
  - NL41::ccgt
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL12::wind_onshore
  - NL11::lost_load
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::ccgt
  - NL22::wind_onshore
  - NL11::pv_farm
  - NL32::lost_load
  - NL32::pv_farm
  - NL33::pv_rooftop
  - NL31::ccgt
  - NL23::lost_load
  - NL13::lost_load
  - NL42::ccgt
  - NL23::ccgt
  - NL21::wind_onshore
  - NL32::pv_rooftop
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL33::pv_farm
  - NL11::wind_offshore
  - NL34::lost_load
  - NL42::lost_load
  - NL13::pv_farm
  - NL42::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL33::wind_offshore
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL41::lost_load
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL32::wind_onshore
  - NL22::pv_rooftop
  - NL21::lost_load
  - NL11::pv_rooftop
  - NL23::wind_offshore
  - NL23::pv_farm
  - NL42::pv_rooftop
  - NL42::wind_onshore
  - NL34::pv_farm
  - NL22::lost_load
  loc_techs_om_cost_supply_plus: []
  loc_techs_out_2: []
  loc_techs_out_3: []
  loc_techs_purchase: []
  loc_techs_ramping:
  - NL42::ccgt
  - NL23::ccgt
  - NL21::ccgt
  - NL12::ccgt
  - NL11::ccgt
  - NL32::ccgt
  - NL41::ccgt
  - NL34::ccgt
  - NL22::ccgt
  - NL13::ccgt
  - NL33::ccgt
  - NL31::ccgt
  loc_techs_storage:
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL12::battery
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL12::hydrogen_to_power
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL23::battery
  - NL31::hydrogen_to_power
  - NL22::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL31::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  loc_techs_store:
  - NL22::battery
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL12::battery
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL41::battery
  - NL12::hydrogen_to_power
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL31::battery
  - NL23::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  - NL31::hydrogen_to_power
  loc_techs_supply:
  - NL41::pv_farm
  - NL21::ccgt
  - NL23::wind_onshore
  - NL32::wind_offshore
  - NL31::wind_onshore
  - NL12::pv_rooftop
  - NL41::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL23::pv_rooftop
  - NL13::ccgt
  - NL34::pv_rooftop
  - NL41::ccgt
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL12::wind_onshore
  - NL11::lost_load
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::ccgt
  - NL22::wind_onshore
  - NL11::pv_farm
  - NL32::lost_load
  - NL32::pv_farm
  - NL33::pv_rooftop
  - NL31::ccgt
  - NL23::lost_load
  - NL13::lost_load
  - NL42::ccgt
  - NL23::ccgt
  - NL21::wind_onshore
  - NL32::pv_rooftop
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL33::pv_farm
  - NL11::wind_offshore
  - NL34::lost_load
  - NL42::pv_rooftop
  - NL42::lost_load
  - NL13::pv_farm
  - NL42::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL33::wind_offshore
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL41::lost_load
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL32::wind_onshore
  - NL22::pv_rooftop
  - NL21::lost_load
  - NL11::pv_rooftop
  - NL23::wind_offshore
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  - NL22::lost_load
  loc_techs_supply_all:
  - NL41::pv_farm
  - NL21::ccgt
  - NL23::wind_onshore
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL23::pv_rooftop
  - NL41::ccgt
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL13::lost_load
  - NL42::ccgt
  - NL23::ccgt
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL11::wind_offshore
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::wind_onshore
  - NL21::lost_load
  - NL23::wind_offshore
  - NL42::pv_rooftop
  - NL22::lost_load
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL34::pv_rooftop
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL21::wind_onshore
  - NL33::pv_farm
  - NL34::lost_load
  - NL42::lost_load
  - NL42::pv_farm
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL34::wind_offshore
  - NL11::ccgt
  - NL33::wind_onshore
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL11::pv_rooftop
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_supply_conversion_all:
  - NL41::pv_farm
  - NL21::ccgt
  - NL23::wind_onshore
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL23::pv_rooftop
  - NL41::ccgt
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL13::lost_load
  - NL42::ccgt
  - NL23::ccgt
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL11::wind_offshore
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::wind_onshore
  - NL21::lost_load
  - NL23::wind_offshore
  - NL42::pv_rooftop
  - NL22::lost_load
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL34::pv_rooftop
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL21::wind_onshore
  - NL33::pv_farm
  - NL34::lost_load
  - NL42::lost_load
  - NL42::pv_farm
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL34::wind_offshore
  - NL11::ccgt
  - NL33::wind_onshore
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL11::pv_rooftop
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_supply_plus: []
  loc_techs_transmission:
  - NL13::hvac_transmission:NL21
  - NL23::hvac_transmission:NL21
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL23::hvac_transmission:NL12
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL11::hvac_transmission:NL13
  - NL41::hvac_transmission:NL42
  - NL21::hvac_transmission:NL13
  - NL22::hvac_transmission:NL21
  - NL41::hvac_transmission:NL34
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL22::hvac_transmission:NL41
  - NL33::hvac_transmission:NL41
  - NL33::hvac_transmission:NL32
  - NL32::hvac_transmission:NL31
  - NL41::hvac_transmission:NL33
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL41::hvac_transmission:NL22
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL21::hvac_transmission:NL23
  - NL31::hvac_transmission:NL32
constraint_sets:
  loc_carriers_system_balance_constraint:
  - NL11::electricity
  - NL22::electricity
  - NL33::electricity
  - NL23::electricity
  - NL42::electricity
  - NL12::electricity
  - NL21::electricity
  - NL34::electricity
  - NL13::electricity
  - NL31::electricity
  - NL32::electricity
  - NL41::electricity
  loc_techs_balance_supply_constraint:
  - NL41::pv_farm
  - NL23::wind_onshore
  - NL32::wind_offshore
  - NL31::wind_onshore
  - NL12::pv_rooftop
  - NL41::pv_rooftop
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL23::pv_rooftop
  - NL31::pv_farm
  - NL34::pv_rooftop
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL12::wind_onshore
  - NL11::lost_load
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL22::wind_onshore
  - NL11::pv_farm
  - NL32::lost_load
  - NL32::pv_farm
  - NL33::pv_rooftop
  - NL23::lost_load
  - NL13::lost_load
  - NL32::pv_rooftop
  - NL21::wind_onshore
  - NL22::pv_farm
  - NL33::pv_farm
  - NL11::wind_offshore
  - NL34::lost_load
  - NL42::lost_load
  - NL13::pv_farm
  - NL42::pv_farm
  - NL33::lost_load
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL33::wind_offshore
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL41::lost_load
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL32::wind_onshore
  - NL22::pv_rooftop
  - NL21::lost_load
  - NL11::pv_rooftop
  - NL23::wind_offshore
  - NL23::pv_farm
  - NL42::pv_rooftop
  - NL42::wind_onshore
  - NL34::pv_farm
  - NL22::lost_load
  loc_techs_balance_demand_constraint:
  - NL21::demand_electricity
  - NL42::demand_electricity
  - NL33::demand_electricity
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL12::demand_electricity
  - NL34::curtailment
  - NL21::curtailment
  - NL41::curtailment
  - NL42::curtailment
  - NL23::demand_electricity
  - NL34::demand_electricity
  - NL33::curtailment
  - NL22::curtailment
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL31::curtailment
  - NL32::curtailment
  - NL31::demand_electricity
  - NL32::demand_electricity
  - NL12::curtailment
  - NL11::curtailment
  - NL13::curtailment
  - NL23::curtailment
  loc_techs_resource_availability_supply_plus_constraint: []
  loc_techs_balance_transmission_constraint:
  - NL13::hvac_transmission:NL21
  - NL23::hvac_transmission:NL21
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL23::hvac_transmission:NL12
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL11::hvac_transmission:NL13
  - NL41::hvac_transmission:NL42
  - NL21::hvac_transmission:NL13
  - NL22::hvac_transmission:NL21
  - NL41::hvac_transmission:NL34
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL22::hvac_transmission:NL41
  - NL33::hvac_transmission:NL41
  - NL33::hvac_transmission:NL32
  - NL32::hvac_transmission:NL31
  - NL41::hvac_transmission:NL33
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL41::hvac_transmission:NL22
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL21::hvac_transmission:NL23
  - NL31::hvac_transmission:NL32
  loc_techs_balance_supply_plus_constraint: []
  loc_techs_balance_storage_constraint:
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL12::battery
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL12::hydrogen_to_power
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL23::battery
  - NL31::hydrogen_to_power
  - NL22::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL31::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  loc_techs_storage_initial_constraint:
  - NL22::battery
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL12::battery
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL41::battery
  - NL12::hydrogen_to_power
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL31::battery
  - NL23::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  - NL31::hydrogen_to_power
  loc_techs_storage_discharge_depth: []
  carriers_reserve_margin_constraint: []
  loc_techs_cost_constraint:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL34::curtailment
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL21::hvac_transmission:NL22
  - NL42::hvac_transmission:NL41
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL41::hvac_transmission:NL42
  - NL21::hvac_transmission:NL13
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL41::hvac_transmission:NL34
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL11::wind_offshore
  - NL33::hvac_transmission:NL41
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL22::lost_load
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL33::hvac_transmission:NL31
  - NL23::hvac_transmission:NL12
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL22::hvac_transmission:NL41
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL42::wind_onshore
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL32::hvac_transmission:NL31
  - NL22::pv_rooftop
  - NL41::hvac_transmission:NL33
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL23::curtailment
  - NL34::pv_farm
  loc_techs_cost_investment_constraint:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL21::hvac_transmission:NL22
  - NL42::hvac_transmission:NL41
  - NL41::ccgt
  - NL12::wind_onshore
  - NL12::wind_offshore
  - NL41::hvac_transmission:NL42
  - NL12::pv_farm
  - NL23::battery
  - NL21::hvac_transmission:NL13
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL31::hydrogen_to_power
  - NL41::hvac_transmission:NL34
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL32::hvac_transmission:NL33
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL23
  - NL11::wind_offshore
  - NL33::hvac_transmission:NL41
  - NL13::pv_farm
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::wind_onshore
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hvac_transmission:NL12
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::pv_farm
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL22::hvac_transmission:NL41
  - NL42::pv_farm
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL34::wind_offshore
  - NL11::ccgt
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL32::hvac_transmission:NL31
  - NL34::ccgt
  - NL41::hvac_transmission:NL33
  - NL22::pv_rooftop
  - NL22::hydrogen_to_power
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_techs_cost_var_constraint:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL34::curtailment
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL22::wind_onshore
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL23::curtailment
  - NL11::wind_offshore
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL22::lost_load
  - NL33::battery
  - NL32::wind_offshore
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL33::wind_onshore
  - NL31::battery
  - NL31::pv_rooftop
  - NL12::lost_load
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  loc_carriers_update_system_balance_constraint: []
  loc_tech_carriers_export_balance_constraint: []
  loc_techs_update_costs_var_constraint: []
  loc_tech_carriers_export_max_constraint: []
  loc_techs_storage_capacity_constraint:
  - NL22::battery
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL12::battery
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL41::battery
  - NL12::hydrogen_to_power
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL31::battery
  - NL23::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  - NL31::hydrogen_to_power
  loc_techs_energy_capacity_storage_constraint_old: []
  loc_techs_energy_capacity_storage_equals_constraint: []
  loc_techs_energy_capacity_storage_min_constraint: []
  loc_techs_energy_capacity_storage_max_constraint:
  - NL22::battery
  - NL33::battery
  - NL12::battery
  - NL11::battery
  - NL32::battery
  - NL34::battery
  - NL41::battery
  - NL42::battery
  - NL21::battery
  - NL31::battery
  - NL23::battery
  - NL13::battery
  loc_techs_resource_capacity_constraint: []
  loc_techs_resource_capacity_equals_energy_capacity_constraint: []
  loc_techs_resource_area_constraint: []
  loc_techs_resource_area_per_energy_capacity_constraint: []
  locs_resource_area_capacity_per_loc_constraint: []
  loc_techs_energy_capacity_constraint:
  - NL41::pv_farm
  - NL11::hydrogen_to_power
  - NL21::ccgt
  - NL23::wind_onshore
  - NL12::battery
  - NL31::wind_onshore
  - NL41::pv_rooftop
  - NL32::battery
  - NL23::pv_rooftop
  - NL13::demand_electricity
  - NL11::demand_electricity
  - NL34::battery
  - NL34::curtailment
  - NL42::hydrogen_to_power
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL41::curtailment
  - NL41::ccgt
  - NL12::wind_onshore
  - NL42::curtailment
  - NL12::wind_offshore
  - NL12::pv_farm
  - NL23::battery
  - NL41::hvac_transmission:NL42
  - NL23::demand_electricity
  - NL22::wind_onshore
  - NL21::hvac_transmission:NL13
  - NL32::pv_farm
  - NL31::ccgt
  - NL23::lost_load
  - NL31::hydrogen_to_power
  - NL13::lost_load
  - NL22::battery
  - NL23::ccgt
  - NL42::ccgt
  - NL41::hvac_transmission:NL34
  - NL12::ccgt
  - NL22::pv_farm
  - NL32::ccgt
  - NL21::hydrogen_to_power
  - NL32::hvac_transmission:NL33
  - NL23::curtailment
  - NL11::wind_offshore
  - NL32::hvac_transmission:NL23
  - NL33::hvac_transmission:NL41
  - NL41::demand_electricity
  - NL22::demand_electricity
  - NL13::pv_farm
  - NL33::lost_load
  - NL33::ccgt
  - NL41::battery
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL33::wind_offshore
  - NL41::lost_load
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL32::demand_electricity
  - NL32::wind_onshore
  - NL12::curtailment
  - NL21::lost_load
  - NL41::hydrogen_to_power
  - NL23::wind_offshore
  - NL41::hvac_transmission:NL22
  - NL13::curtailment
  - NL42::pv_rooftop
  - NL31::hvac_transmission:NL32
  - NL22::lost_load
  - NL33::battery
  - NL13::hvac_transmission:NL21
  - NL21::demand_electricity
  - NL32::wind_offshore
  - NL42::demand_electricity
  - NL12::pv_rooftop
  - NL32::hydrogen_to_power
  - NL34::wind_onshore
  - NL33::demand_electricity
  - NL21::pv_rooftop
  - NL13::wind_onshore
  - NL13::ccgt
  - NL23::hvac_transmission:NL21
  - NL12::demand_electricity
  - NL12::hydrogen_to_power
  - NL34::pv_rooftop
  - NL23::hvac_transmission:NL12
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL21::curtailment
  - NL11::hvac_transmission:NL13
  - NL13::pv_rooftop
  - NL41::wind_onshore
  - NL22::ccgt
  - NL11::lost_load
  - NL11::pv_farm
  - NL32::lost_load
  - NL33::pv_rooftop
  - NL22::hvac_transmission:NL21
  - NL32::pv_rooftop
  - NL34::hydrogen_to_power
  - NL21::wind_onshore
  - NL21::hvac_transmission:NL23
  - NL34::demand_electricity
  - NL11::battery
  - NL33::pv_farm
  - NL33::hydrogen_to_power
  - NL34::lost_load
  - NL33::curtailment
  - NL42::lost_load
  - NL22::curtailment
  - NL22::hvac_transmission:NL41
  - NL32::hvac_transmission:NL31
  - NL42::pv_farm
  - NL31::curtailment
  - NL31::lost_load
  - NL33::hvac_transmission:NL32
  - NL11::wind_onshore
  - NL21::pv_farm
  - NL32::curtailment
  - NL11::ccgt
  - NL34::wind_offshore
  - NL31::demand_electricity
  - NL33::wind_onshore
  - NL31::battery
  - NL12::lost_load
  - NL31::pv_rooftop
  - NL34::ccgt
  - NL22::pv_rooftop
  - NL41::hvac_transmission:NL33
  - NL22::hydrogen_to_power
  - NL11::curtailment
  - NL11::pv_rooftop
  - NL13::battery
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL23::pv_farm
  - NL31::pv_farm
  - NL42::wind_onshore
  - NL34::pv_farm
  techs_energy_capacity_systemwide_constraint:
  - ccgt
  loc_tech_carriers_carrier_production_max_constraint:
  - NL33::battery::electricity
  - NL22::pv_farm::electricity
  - NL42::hydrogen_to_power::electricity
  - NL22::battery::electricity
  - NL42::pv_rooftop::electricity
  - NL21::hydrogen_to_power::electricity
  - NL41::hvac_transmission:NL33::electricity
  - NL32::ccgt::electricity
  - NL13::pv_rooftop::electricity
  - NL22::hvac_transmission:NL21::electricity
  - NL21::wind_onshore::electricity
  - NL41::battery::electricity
  - NL23::battery::electricity
  - NL12::battery::electricity
  - NL32::wind_offshore::electricity
  - NL31::hvac_transmission:NL32::electricity
  - NL12::hvac_transmission:NL23::electricity
  - NL21::hvac_transmission:NL13::electricity
  - NL12::pv_farm::electricity
  - NL12::hydrogen_to_power::electricity
  - NL32::hydrogen_to_power::electricity
  - NL13::hvac_transmission:NL11::electricity
  - NL13::hvac_transmission:NL21::electricity
  - NL34::wind_offshore::electricity
  - NL41::hvac_transmission:NL42::electricity
  - NL32::hvac_transmission:NL31::electricity
  - NL34::pv_rooftop::electricity
  - NL21::battery::electricity
  - NL41::wind_onshore::electricity
  - NL12::ccgt::electricity
  - NL34::hydrogen_to_power::electricity
  - NL21::lost_load::electricity
  - NL32::battery::electricity
  - NL34::pv_farm::electricity
  - NL11::hvac_transmission:NL12::electricity
  - NL41::hvac_transmission:NL34::electricity
  - NL33::wind_offshore::electricity
  - NL21::hvac_transmission:NL22::electricity
  - NL41::hydrogen_to_power::electricity
  - NL13::lost_load::electricity
  - NL33::lost_load::electricity
  - NL42::ccgt::electricity
  - NL12::wind_onshore::electricity
  - NL21::pv_farm::electricity
  - NL41::ccgt::electricity
  - NL12::wind_offshore::electricity
  - NL11::wind_onshore::electricity
  - NL31::battery::electricity
  - NL11::pv_rooftop::electricity
  - NL23::lost_load::electricity
  - NL31::ccgt::electricity
  - NL41::pv_farm::electricity
  - NL42::wind_onshore::electricity
  - NL33::ccgt::electricity
  - NL31::wind_onshore::electricity
  - NL22::ccgt::electricity
  - NL33::hvac_transmission:NL31::electricity
  - NL11::hvac_transmission:NL13::electricity
  - NL22::pv_rooftop::electricity
  - NL11::wind_offshore::electricity
  - NL31::hydrogen_to_power::electricity
  - NL41::lost_load::electricity
  - NL23::pv_farm::electricity
  - NL11::lost_load::electricity
  - NL33::hydrogen_to_power::electricity
  - NL31::pv_rooftop::electricity
  - NL33::hvac_transmission:NL32::electricity
  - NL32::hvac_transmission:NL33::electricity
  - NL22::hydrogen_to_power::electricity
  - NL22::lost_load::electricity
  - NL32::lost_load::electricity
  - NL33::hvac_transmission:NL41::electricity
  - NL22::wind_onshore::electricity
  - NL41::pv_rooftop::electricity
  - NL33::wind_onshore::electricity
  - NL12::pv_rooftop::electricity
  - NL33::pv_rooftop::electricity
  - NL11::pv_farm::electricity
  - NL23::pv_rooftop::electricity
  - NL23::ccgt::electricity
  - NL11::battery::electricity
  - NL32::pv_farm::electricity
  - NL41::hvac_transmission:NL22::electricity
  - NL34::wind_onshore::electricity
  - NL32::hvac_transmission:NL23::electricity
  - NL23::hvac_transmission:NL32::electricity
  - NL21::ccgt::electricity
  - NL13::ccgt::electricity
  - NL31::hvac_transmission:NL33::electricity
  - NL34::battery::electricity
  - NL23::wind_offshore::electricity
  - NL34::lost_load::electricity
  - NL21::hvac_transmission:NL23::electricity
  - NL11::ccgt::electricity
  - NL31::pv_farm::electricity
  - NL13::hydrogen_to_power::electricity
  - NL34::hvac_transmission:NL41::electricity
  - NL42::hvac_transmission:NL41::electricity
  - NL42::lost_load::electricity
  - NL12::hvac_transmission:NL11::electricity
  - NL13::wind_onshore::electricity
  - NL34::ccgt::electricity
  - NL12::lost_load::electricity
  - NL13::battery::electricity
  - NL23::hydrogen_to_power::electricity
  - NL23::hvac_transmission:NL21::electricity
  - NL31::lost_load::electricity
  - NL21::pv_rooftop::electricity
  - NL13::pv_farm::electricity
  - NL23::wind_onshore::electricity
  - NL32::pv_rooftop::electricity
  - NL32::wind_onshore::electricity
  - NL11::hydrogen_to_power::electricity
  - NL33::pv_farm::electricity
  - NL42::pv_farm::electricity
  - NL22::hvac_transmission:NL41::electricity
  - NL23::hvac_transmission:NL12::electricity
  - NL42::battery::electricity
  loc_tech_carriers_carrier_production_min_constraint: []
  loc_tech_carriers_carrier_consumption_max_constraint:
  - NL41::hvac_transmission:NL42::electricity
  - NL41::hvac_transmission:NL22::electricity
  - NL32::hvac_transmission:NL31::electricity
  - NL33::battery::electricity
  - NL21::battery::electricity
  - NL32::hvac_transmission:NL23::electricity
  - NL23::hvac_transmission:NL32::electricity
  - NL13::hvac_transmission:NL11::electricity
  - NL33::curtailment::electricity
  - NL11::curtailment::electricity
  - NL42::hydrogen_to_power::electricity
  - NL22::battery::electricity
  - NL34::demand_electricity::electricity
  - NL11::demand_electricity::electricity
  - NL21::curtailment::electricity
  - NL42::curtailment::electricity
  - NL34::hydrogen_to_power::electricity
  - NL31::hvac_transmission:NL33::electricity
  - NL34::battery::electricity
  - NL21::hydrogen_to_power::electricity
  - NL13::battery::electricity
  - NL32::demand_electricity::electricity
  - NL42::demand_electricity::electricity
  - NL21::hvac_transmission:NL23::electricity
  - NL41::hvac_transmission:NL33::electricity
  - NL33::hvac_transmission:NL31::electricity
  - NL11::hvac_transmission:NL13::electricity
  - NL32::battery::electricity
  - NL31::hydrogen_to_power::electricity
  - NL13::hydrogen_to_power::electricity
  - NL34::hvac_transmission:NL41::electricity
  - NL33::hydrogen_to_power::electricity
  - NL33::demand_electricity::electricity
  - NL42::hvac_transmission:NL41::electricity
  - NL32::hydrogen_to_power::electricity
  - NL11::hvac_transmission:NL12::electricity
  - NL12::hvac_transmission:NL11::electricity
  - NL33::hvac_transmission:NL32::electricity
  - NL41::hvac_transmission:NL34::electricity
  - NL21::demand_electricity::electricity
  - NL22::demand_electricity::electricity
  - NL21::hvac_transmission:NL22::electricity
  - NL23::curtailment::electricity
  - NL41::hydrogen_to_power::electricity
  - NL32::hvac_transmission:NL33::electricity
  - NL13::curtailment::electricity
  - NL23::hvac_transmission:NL21::electricity
  - NL12::demand_electricity::electricity
  - NL23::hydrogen_to_power::electricity
  - NL22::hvac_transmission:NL21::electricity
  - NL31::battery::electricity
  - NL22::hydrogen_to_power::electricity
  - NL31::curtailment::electricity
  - NL41::battery::electricity
  - NL23::battery::electricity
  - NL13::demand_electricity::electricity
  - NL12::battery::electricity
  - NL33::hvac_transmission:NL41::electricity
  - NL23::demand_electricity::electricity
  - NL34::curtailment::electricity
  - NL41::curtailment::electricity
  - NL31::demand_electricity::electricity
  - NL11::hydrogen_to_power::electricity
  - NL12::curtailment::electricity
  - NL31::hvac_transmission:NL32::electricity
  - NL12::hvac_transmission:NL23::electricity
  - NL22::hvac_transmission:NL41::electricity
  - NL21::hvac_transmission:NL13::electricity
  - NL32::curtailment::electricity
  - NL11::battery::electricity
  - NL23::hvac_transmission:NL12::electricity
  - NL12::hydrogen_to_power::electricity
  - NL13::hvac_transmission:NL21::electricity
  - NL42::battery::electricity
  - NL41::demand_electricity::electricity
  - NL22::curtailment::electricity
  loc_techs_resource_max_constraint: []
  loc_tech_carriers_ramping_constraint:
  - NL32::ccgt::electricity
  - NL12::ccgt::electricity
  - NL42::ccgt::electricity
  - NL41::ccgt::electricity
  - NL31::ccgt::electricity
  - NL33::ccgt::electricity
  - NL22::ccgt::electricity
  - NL23::ccgt::electricity
  - NL21::ccgt::electricity
  - NL13::ccgt::electricity
  - NL11::ccgt::electricity
  - NL34::ccgt::electricity
  loc_techs_storage_max_constraint:
  - NL22::battery
  - NL11::hydrogen_to_power
  - NL33::battery
  - NL34::hydrogen_to_power
  - NL21::hydrogen_to_power
  - NL12::battery
  - NL11::battery
  - NL33::hydrogen_to_power
  - NL32::hydrogen_to_power
  - NL32::battery
  - NL42::hydrogen_to_power
  - NL34::battery
  - NL41::battery
  - NL12::hydrogen_to_power
  - NL13::hydrogen_to_power
  - NL42::battery
  - NL23::hydrogen_to_power
  - NL21::battery
  - NL31::battery
  - NL23::battery
  - NL22::hydrogen_to_power
  - NL41::hydrogen_to_power
  - NL13::battery
  - NL31::hydrogen_to_power
  loc_techs_unit_commitment_milp_constraint: []
  loc_techs_unit_capacity_milp_constraint: []
  loc_tech_carriers_carrier_production_max_milp_constraint: []
  loc_techs_carrier_production_max_conversion_plus_milp_constraint: []
  loc_tech_carriers_carrier_production_min_milp_constraint: []
  loc_techs_carrier_production_min_conversion_plus_milp_constraint: []
  loc_tech_carriers_carrier_consumption_max_milp_constraint: []
  loc_techs_energy_capacity_units_milp_constraint: []
  loc_techs_storage_capacity_units_milp_constraint: []
  loc_techs_energy_capacity_max_purchase_milp_constraint: []
  loc_techs_energy_capacity_min_purchase_milp_constraint: []
  loc_techs_storage_capacity_max_purchase_milp_constraint: []
  loc_techs_storage_capacity_min_purchase_milp_constraint: []
  loc_techs_update_costs_investment_units_milp_constraint: []
  loc_techs_update_costs_investment_purchase_milp_constraint: []
  techs_unit_capacity_systemwide_milp_constraint: []
  loc_techs_asynchronous_prod_con_milp_constraint: []
  loc_techs_balance_conversion_constraint: []
  loc_techs_cost_var_conversion_constraint: []
  loc_techs_balance_conversion_plus_primary_constraint: []
  loc_techs_carrier_production_max_conversion_plus_constraint: []
  loc_techs_carrier_production_min_conversion_plus_constraint: []
  loc_techs_cost_var_conversion_plus_constraint: []
  loc_techs_balance_conversion_plus_in_2_constraint: []
  loc_techs_balance_conversion_plus_in_3_constraint: []
  loc_techs_balance_conversion_plus_out_2_constraint: []
  loc_techs_balance_conversion_plus_out_3_constraint: []
  loc_techs_symmetric_transmission_constraint:
  - NL13::hvac_transmission:NL21
  - NL23::hvac_transmission:NL21
  - NL11::hvac_transmission:NL12
  - NL23::hvac_transmission:NL32
  - NL23::hvac_transmission:NL12
  - NL42::hvac_transmission:NL41
  - NL21::hvac_transmission:NL22
  - NL33::hvac_transmission:NL31
  - NL34::hvac_transmission:NL41
  - NL11::hvac_transmission:NL13
  - NL41::hvac_transmission:NL42
  - NL21::hvac_transmission:NL13
  - NL22::hvac_transmission:NL21
  - NL41::hvac_transmission:NL34
  - NL32::hvac_transmission:NL33
  - NL32::hvac_transmission:NL23
  - NL22::hvac_transmission:NL41
  - NL33::hvac_transmission:NL41
  - NL33::hvac_transmission:NL32
  - NL32::hvac_transmission:NL31
  - NL41::hvac_transmission:NL33
  - NL13::hvac_transmission:NL11
  - NL12::hvac_transmission:NL11
  - NL41::hvac_transmission:NL22
  - NL31::hvac_transmission:NL33
  - NL12::hvac_transmission:NL23
  - NL21::hvac_transmission:NL23
  - NL31::hvac_transmission:NL32
  techlists_group_share_energy_cap_min_constraint: []
  techlists_carrier_group_share_carrier_prod_min_constraint: []
  techlists_group_share_energy_cap_max_constraint: []
  techlists_carrier_group_share_carrier_prod_max_constraint: []
  techlists_group_share_energy_cap_equals_constraint: []
  techlists_carrier_group_share_carrier_prod_equals_constraint: []
  group_constraints: []
BTLF �      ��            ��     8�             .[�,                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDRF                                                 �T                  �T           �     .          	 ������������������������0        CLASS                DIMENSION_SCALE &       NAME                 techs 4       _Netcdf4Dimid                             u��OHDRF                                                 �T     4             �T     4      H�     �           ������������������������0        CLASS                DIMENSION_SCALE &       NAME                 costs 4       _Netcdf4Dimid                            �OCOHDRM                                                 �T     7             �T     7      	B     �            ������������������������0        CLASS                DIMENSION_SCALE -       NAME                 loc_carriers 4       _Netcdf4Dimid                            %`��OHDRg                                                 �T     P             �T     P       ��     �            ������������������������0        CLASS                DIMENSION_SCALE G       NAME       '          loc_carriers_system_balance_constraint 4       _Netcdf4Dimid                            ��X   ���BTHD      d(�      h       O�.BTHD      d(2�      h       w�5�FSSE �k       �   �     �	     �   &  �   ��[      _debug_data        �         comments:
  model:
    time:
      function: Applied from override
      function_options:
        resolution: Applied from override
  model_run:
    techs:
      battery:
        essentials:
          parent: From parent tech_group `storage`
          carrier_in: Set from essentials.carrier
      ccgt:
        essentials:
          parent: From parent tech_group `supply`
      curtailment:
        essentials:
          parent: From parent tech_group `demand`
      demand_electricity:
        essentials:
          parent: From parent tech_group `demand`
          carrier_in: Set from essentials.carrier
      export_electricity:
        essentials:
          parent: From parent tech_group `demand`
      hvac_transmission:
        essentials:
          parent: From parent tech_group `transmission`
          carrier_in: Set from essentials.carrier
      hydrogen_to_power:
        essentials:
          parent: From parent tech_group `storage`
          carrier_in: Set from essentials.carrier
      import_electricity:
        essentials:
          parent: From parent tech_group `supply`
      lost_load:
        essentials:
          parent: From parent tech_group `supply`
      pv_farm:
        essentials:
          parent: From parent tech_group `supply`
      pv_rooftop:
        essentials:
          parent: From parent tech_group `supply`
      wind_offshore:
        essentials:
          parent: From parent tech_group `supply`
      wind_onshore:
        essentials:
          parent: From parent tech_group `supply`
    locations: {}
config_initial:
  run:
    backend: pyomo
    bigM: 1000000.0
    cyclic_storage: true
    ensure_feasibility: true
    mode: plan
    objective_options:
      cost_class:
        monetary: 1
      sense: minimize
    objective: minmax_cost_optimization
    operation:
      window:
      horizon:
      use_cap_results: false
    spores_options:
      spores_number: 3
      slack: 0.1
      score_cost_class: spores_score
      objective_cost_class:
        spores_score: 1
        monetary: 0
      slack_cost_group:
      save_per_spore: false
      save_per_spore_path:
      skip_cost_op: false
    relax_constraint:
      demand_share_per_timestep_decision_main_constraint: 0
    save_logs:
    solver_io:
    solver_options:
    solver: cbc
    zero_threshold: 1e-10
  model:
    calliope_version: 0.6.10
    group_share: {}
    name: Dutch power system at NUTS2 resolution
    random_seed:
    reserve_margin: {}
    subset_time:
    - '2019-01-01'
    - '2019-01-31'
    time:
      function: resample
      function_options:
        resolution: 3H
    timeseries_data_path: model_files/timeseries_data
    timeseries_data:
    timeseries_dateformat: '%Y-%m-%d %H:%M:%S'
    file_allowed:
    - clustering_func
    - energy_con
    - energy_eff
    - energy_prod
    - energy_ramping
    - export
    - force_resource
    - om_con
    - om_prod
    - parasitic_eff
    - resource
    - resource_eff
    - storage_loss
    - carrier_ratios
  tech_groups:
    conversion:
      required_constraints: []
      allowed_constraints:
      - energy_cap_equals
      - energy_cap_equals_systemwide
      - energy_cap_max
      - energy_cap_max_systemwide
      - energy_cap_min
      - energy_cap_min_use
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_con
      - energy_eff
      - energy_prod
      - energy_ramping
      - export_cap
      - export_carrier
      - lifetime
      - units_equals
      - units_equals_systemwide
      - units_max
      - units_max_systemwide
      - units_min
      allowed_group_constraints:
      - demand_share_min
      - demand_share_max
      - demand_share_equals
      - demand_share_per_timestep_min
      - demand_share_per_timestep_max
      - demand_share_per_timestep_equals
      - demand_share_per_timestep_decision
      - carrier_prod_share_min
      - carrier_prod_share_max
      - carrier_prod_share_equals
      - carrier_prod_share_per_timestep_min
      - carrier_prod_share_per_timestep_max
      - carrier_prod_share_per_timestep_equals
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_share_min
      - energy_cap_share_max
      - energy_cap_share_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - carrier_prod_min
      - carrier_prod_max
      - carrier_prod_equals
      - carrier_con_min
      - carrier_con_max
      - carrier_con_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - export
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_con
      - om_prod
      - purchase
      essentials:
        parent:
      constraints:
        energy_con: true
        energy_prod: true
    conversion_plus:
      required_constraints: []
      allowed_constraints:
      - carrier_ratios
      - energy_cap_equals
      - energy_cap_equals_systemwide
      - energy_cap_max
      - energy_cap_max_systemwide
      - energy_cap_min
      - energy_cap_min_use
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_con
      - energy_eff
      - energy_prod
      - energy_ramping
      - export_cap
      - export_carrier
      - lifetime
      - units_equals
      - units_equals_systemwide
      - units_max
      - units_max_systemwide
      - units_min
      allowed_group_constraints:
      - demand_share_min
      - demand_share_max
      - demand_share_equals
      - demand_share_per_timestep_min
      - demand_share_per_timestep_max
      - demand_share_per_timestep_equals
      - demand_share_per_timestep_decision
      - carrier_prod_share_min
      - carrier_prod_share_max
      - carrier_prod_share_equals
      - carrier_prod_share_per_timestep_min
      - carrier_prod_share_per_timestep_max
      - carrier_prod_share_per_timestep_equals
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_share_min
      - energy_cap_share_max
      - energy_cap_share_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - carrier_prod_min
      - carrier_prod_max
      - carrier_prod_equals
      - carrier_con_min
      - carrier_con_max
      - carrier_con_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - export
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_con
      - om_prod
      - purchase
      essentials:
        parent:
      constraints:
        energy_con: true
        energy_prod: true
    demand:
      required_constraints:
      - resource
      allowed_constraints:
      - energy_con
      - force_resource
      - resource
      - resource_area_equals
      - resource_scale
      - resource_unit
      allowed_group_constraints:
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - carrier_con_min
      - carrier_con_max
      - carrier_con_equals
      allowed_costs:
      - om_con
      essentials:
        parent:
      constraints:
        energy_con: true
        force_resource: true
        resource_unit: energy
    storage:
      required_constraints: []
      allowed_constraints:
      - charge_rate
      - energy_cap_per_storage_cap_min
      - energy_cap_per_storage_cap_max
      - energy_cap_per_storage_cap_equals
      - energy_cap_equals
      - energy_cap_equals_systemwide
      - energy_cap_max
      - energy_cap_max_systemwide
      - energy_cap_min
      - energy_cap_min_use
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_con
      - energy_eff
      - energy_prod
      - energy_ramping
      - export_cap
      - export_carrier
      - force_asynchronous_prod_con
      - lifetime
      - storage_cap_equals
      - storage_cap_max
      - storage_cap_min
      - storage_cap_per_unit
      - storage_initial
      - storage_loss
      - storage_time_max
      - storage_discharge_depth
      - units_equals
      - units_equals_systemwide
      - units_max
      - units_max_systemwide
      - units_min
      allowed_group_constraints:
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_share_min
      - energy_cap_share_max
      - energy_cap_share_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - storage_cap_min
      - storage_cap_max
      - storage_cap_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - export
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_prod
      - purchase
      - storage_cap
      essentials:
        parent:
      constraints:
        energy_con: true
        energy_prod: true
        storage_cap_max: inf
    supply:
      required_constraints: []
      allowed_constraints:
      - energy_cap_equals
      - energy_cap_equals_systemwide
      - energy_cap_max
      - energy_cap_max_systemwide
      - energy_cap_min
      - energy_cap_min_use
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_eff
      - energy_prod
      - energy_ramping
      - export_cap
      - export_carrier
      - force_resource
      - lifetime
      - resource
      - resource_area_equals
      - resource_area_max
      - resource_area_min
      - resource_area_per_energy_cap
      - resource_min_use
      - resource_scale
      - resource_unit
      - units_equals
      - units_equals_systemwide
      - units_max
      - units_max_systemwide
      - units_min
      allowed_group_constraints:
      - demand_share_min
      - demand_share_max
      - demand_share_equals
      - demand_share_per_timestep_min
      - demand_share_per_timestep_max
      - demand_share_per_timestep_equals
      - demand_share_per_timestep_decision
      - carrier_prod_share_min
      - carrier_prod_share_max
      - carrier_prod_share_equals
      - carrier_prod_share_per_timestep_min
      - carrier_prod_share_per_timestep_max
      - carrier_prod_share_per_timestep_equals
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_share_min
      - energy_cap_share_max
      - energy_cap_share_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - resource_area_min
      - resource_area_max
      - resource_area_equals
      - carrier_prod_min
      - carrier_prod_max
      - carrier_prod_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - export
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_con
      - om_prod
      - purchase
      - resource_area
      essentials:
        parent:
      constraints:
        energy_prod: true
        resource: inf
        resource_unit: energy
    supply_plus:
      required_constraints: []
      allowed_constraints:
      - charge_rate
      - energy_cap_per_storage_cap_min
      - energy_cap_per_storage_cap_max
      - energy_cap_per_storage_cap_equals
      - energy_cap_equals
      - energy_cap_equals_systemwide
      - energy_cap_max
      - energy_cap_max_systemwide
      - energy_cap_min
      - energy_cap_min_use
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_eff
      - energy_prod
      - energy_ramping
      - export_cap
      - export_carrier
      - force_resource
      - lifetime
      - parasitic_eff
      - resource
      - resource_area_equals
      - resource_area_max
      - resource_area_min
      - resource_area_per_energy_cap
      - resource_cap_equals
      - resource_cap_equals_energy_cap
      - resource_cap_max
      - resource_cap_min
      - resource_eff
      - resource_min_use
      - resource_scale
      - resource_unit
      - storage_cap_equals
      - storage_cap_max
      - storage_cap_min
      - storage_cap_per_unit
      - storage_initial
      - storage_loss
      - units_equals
      - units_equals_systemwide
      - units_max
      - units_max_systemwide
      - units_min
      allowed_group_constraints:
      - demand_share_min
      - demand_share_max
      - demand_share_equals
      - demand_share_per_timestep_min
      - demand_share_per_timestep_max
      - demand_share_per_timestep_equals
      - demand_share_per_timestep_decision
      - carrier_prod_share_min
      - carrier_prod_share_max
      - carrier_prod_share_equals
      - carrier_prod_share_per_timestep_min
      - carrier_prod_share_per_timestep_max
      - carrier_prod_share_per_timestep_equals
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_share_min
      - energy_cap_share_max
      - energy_cap_share_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - resource_area_min
      - resource_area_max
      - resource_area_equals
      - carrier_prod_min
      - carrier_prod_max
      - carrier_prod_equals
      - storage_cap_min
      - storage_cap_max
      - storage_cap_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - export
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_con
      - om_prod
      - purchase
      - resource_area
      - resource_cap
      - storage_cap
      essentials:
        parent:
      constraints:
        energy_prod: true
        resource: inf
        resource_eff: 1.0
        resource_unit: energy
    transmission:
      required_constraints: []
      allowed_constraints:
      - energy_cap_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_per_unit
      - energy_cap_scale
      - energy_con
      - energy_eff
      - energy_eff_per_distance
      - energy_prod
      - force_asynchronous_prod_con
      - lifetime
      - one_way
      allowed_group_constraints:
      - cost_max
      - cost_min
      - cost_equals
      - cost_var_max
      - cost_var_min
      - cost_var_equals
      - cost_investment_min
      - cost_investment_max
      - cost_investment_equals
      - energy_cap_min
      - energy_cap_max
      - energy_cap_equals
      - net_import_share_min
      - net_import_share_max
      - net_import_share_equals
      allowed_costs:
      - depreciation_rate
      - energy_cap
      - energy_cap_per_distance
      - interest_rate
      - om_annual
      - om_annual_investment_fraction
      - om_prod
      - purchase
      - purchase_per_distance
      essentials:
        parent:
      constraints:
        energy_con: true
        energy_prod: true
  techs:
    battery:
      constraints:
        energy_cap_per_storage_cap_max: 0.25
        energy_eff: 0.98
        lifetime: 22.5
        storage_initial: 0
      costs:
        monetary:
          interest_rate: 0.1
          om_annual_investment_fraction: 0.57
          om_prod: 0.002
          storage_cap: 885
      essentials:
        carrier: electricity
        color: '#33F6FF'
        name: Battery storage
        parent: storage
    ccgt:
      constraints:
        energy_cap_max_systemwide: 10000
        energy_eff: 0.57
        energy_ramping: 0.8
        lifetime: 25
        resource: inf
      costs:
        monetary:
          energy_cap: 904.7795
          interest_rate: 0.1
          om_annual_investment_fraction: 3.3392
          om_con: 0.1
          om_prod: 0.00455
      essentials:
        carrier_out: electricity
        color: '#AB2C18'
        name: Combined cycle gas turbine
        parent: supply
    curtailment:
      constraints:
        force_resource: false
        resource: -100000.0
      costs:
        monetary:
          om_con: 0
      essentials:
        carrier_in: electricity
        color: '#6F6C7C'
        name: Electricity curtailment
        parent: demand
    demand_electricity:
      constraints:
        force_resource: true
        resource: file=electricity_demand_nuts2.csv
      essentials:
        carrier: electricity
        color: '#072486'
        name: Electricity demand
        parent: demand
    export_electricity:
      constraints:
        force_resource: false
        resource: -100000.0
      costs:
        monetary:
          om_con: -0.1
      essentials:
        carrier_in: electricity
        color: '#D800FF'
        name: Electricity export
        parent: demand
    hvac_transmission:
      constraints:
        energy_eff: 0.96
        lifetime: 60
      costs:
        monetary:
          energy_cap: 175
          interest_rate: 0.1
          om_annual_investment_fraction: 0.015
      essentials:
        carrier: electricity
        color: '#CF74B2'
        name: Transmission intra-zonal electricity
        parent: transmission
    hydrogen_to_power:
      constraints:
        energy_eff: 0.542
        lifetime: 15
        storage_initial: 0
      costs:
        monetary:
          energy_cap: 3069.866
          interest_rate: 0.1
          om_annual_investment_fraction: 4
          om_prod: 0
          storage_cap: 22.982
      essentials:
        carrier: electricity
        color: '#9958D6'
        name: Hydrogen-to-power storage
        parent: storage
    import_electricity:
      constraints:
        energy_prod: true
        force_resource: false
        lifetime: 40
        resource: 100000.0
        resource_eff: 1
      costs:
        monetary:
          om_prod: 0.1
      essentials:
        carrier_out: electricity
        color: '#FF009E'
        name: Electricity import
        parent: supply
    lost_load:
      constraints:
        energy_prod: true
        force_resource: false
        lifetime: 20
        resource: 100000.0
      costs:
        monetary:
          om_prod: 10000.0
      essentials:
        carrier_out: electricity
        color: '#030303'
        name: Lost load
        parent: supply
    pv_farm:
      constraints:
        force_resource: true
        lifetime: 37.5
        resource: file=pv_nuts2.csv
        resource_unit: energy_per_cap
      costs:
        monetary:
          energy_cap: 676.5703
          interest_rate: 0.1
          om_annual_investment_fraction: 1.7275
          om_prod: 0
      essentials:
        carrier_out: electricity
        color: '#FFE400'
        name: Photovoltaic power farm-scale
        parent: supply
    pv_rooftop:
      constraints:
        force_resource: true
        lifetime: 37.5
        resource: file=pv_nuts2.csv
        resource_unit: energy_per_cap
      costs:
        monetary:
          energy_cap: 880.0251
          interest_rate: 0.1
          om_annual_investment_fraction: 1.2567
          om_prod: 0
      essentials:
        carrier_out: electricity
        color: '#FFB600'
        name: Photovoltaic power distributed
        parent: supply
    wind_offshore:
      constraints:
        force_resource: true
        lifetime: 30
        resource: file=windoff_nuts2.csv
        resource_unit: energy_per_cap
      costs:
        monetary:
          energy_cap: 1139.8826
          interest_rate: 0.1
          om_annual_investment_fraction: 2.3741
          om_prod: 0
      essentials:
        carrier_out: electricity
        color: '#275FB0'
        name: Off-shore wind power
        parent: supply
    wind_onshore:
      constraints:
        force_resource: true
        lifetime: 28.5
        resource: file=wind_nuts2.csv
        resource_unit: energy_per_cap
      costs:
        monetary:
          energy_cap: 1139.8826
          interest_rate: 0.1
          om_annual_investment_fraction: 1.2347
          om_prod: 0
      essentials:
        carrier_out: electricity
        color: '#27B09F'
        name: On-shore wind power
        parent: supply
  locations:
    NL11:
      coordinates:
        lat: 53.151875
        lon: 6.8779375
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL12:
      coordinates:
        lat: 53.130320000000005
        lon: 5.94212
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL13:
      coordinates:
        lat: 52.79966
        lon: 6.69054
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL21:
      coordinates:
        lat: 52.41295
        lon: 6.442433333333334
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL22:
      coordinates:
        lat: 51.94764
        lon: 6.012
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL23:
      coordinates:
        lat: 52.60066666666666
        lon: 5.672633333333333
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL31:
      coordinates:
        lat: 52.1608
        lon: 4.9873
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL32:
      coordinates:
        lat: 52.36538333333333
        lon: 4.748033333333333
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL33:
      coordinates:
        lat: 51.95356666666667
        lon: 4.452383333333334
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL34:
      coordinates:
        lat: 51.4261
        lon: 4.2185
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_offshore:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL41:
      coordinates:
        lat: 51.573025
        lon: 5.135325
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
    NL42:
      coordinates:
        lat: 51.1524
        lon: 5.9052
      techs:
        battery:
        ccgt:
        curtailment:
        demand_electricity:
        hydrogen_to_power:
        lost_load:
        pv_farm:
          constraints:
            energy_cap_max: 2500
        pv_rooftop:
          constraints:
            energy_cap_max: 5000
        wind_onshore:
          constraints:
            energy_cap_max: 5000
  links:
    NL11,NL12:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 3.4401
            energy_cap_min: 0.6882
          distance: 78.08059088268635
    NL11,NL13:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 15.3276
            energy_cap_min: 3.0655
          distance: 51.40669619092387
    NL12,NL23:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 3.4401
            energy_cap_min: 0.6882
          distance: 77.01279250805582
    NL13,NL21:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 15.3276
            energy_cap_min: 3.0655
          distance: 57.68592103711268
    NL21,NL22:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 74.3546324021931
    NL21,NL23:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 15.3276
            energy_cap_min: 3.0655
          distance: 70.1574152900598
    NL22,NL41:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 91.64766157833785
    NL23,NL32:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 16.9811
            energy_cap_min: 3.3962
          distance: 84.8212339513288
    NL31,NL32:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 34.96977818838089
    NL31,NL33:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 1.1886
            energy_cap_min: 0.2377
          distance: 54.0333446482031
    NL32,NL33:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 62.54522684034896
    NL33,NL41:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 79.05131602490407
    NL34,NL41:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 11.8867
            energy_cap_min: 2.3773
          distance: 81.91534737849409
    NL41,NL42:
      techs:
        hvac_transmission:
          constraints:
            energy_cap_max: 23.7734
            energy_cap_min: 4.7547
          distance: 88.7810048549968
  config_path: model_files/model.yaml
  overrides:
    fix-design-to-cost-optimal:
      links:
        NL11,NL12:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 3.4401
        NL11,NL13:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 15.3276
        NL12,NL23:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 3.4401
        NL13,NL21:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 15.3276
        NL21,NL22:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL21,NL23:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 15.3276
        NL22,NL41:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL23,NL32:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 16.9811
        NL31,NL32:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL31,NL33:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 1.1886
        NL32,NL33:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL33,NL41:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL34,NL41:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 11.8867
        NL41,NL42:
          techs:
            hvac_transmission:
              constraints:
                energy_cap_equals: 23.7734
      locations:
        NL11:
          techs:
            battery:
              constraints:
                energy_cap_equals: 127.09184400594432
            ccgt:
              constraints:
                energy_cap_equals: 304.2295415419153
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 140.3362102851781
            pv_farm:
              constraints:
                energy_cap_equals: 629.8406369363555
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_offshore:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 1428.2440610613542
        NL12:
          techs:
            battery:
              constraints:
                energy_cap_equals: 99.12113807415821
            ccgt:
              constraints:
                energy_cap_equals: 313.7861763388139
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 55.92251784096582
            pv_farm:
              constraints:
                energy_cap_equals: 511.61749475334585
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_offshore:
              constraints:
                energy_cap_equals: 136.37459875649907
            wind_onshore:
              constraints:
                energy_cap_equals: 698.2304212057271
        NL13:
          techs:
            battery:
              constraints:
                energy_cap_equals: 107.19872910502045
            ccgt:
              constraints:
                energy_cap_equals: 215.02310998453098
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 36.01293725312823
            pv_farm:
              constraints:
                energy_cap_equals: 426.3130139070597
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 735.1239854261394
        NL21:
          techs:
            battery:
              constraints:
                energy_cap_equals: 436.37272036523007
            ccgt:
              constraints:
                energy_cap_equals: 527.6226293713593
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 155.2622654135686
            pv_farm:
              constraints:
                energy_cap_equals: 1775.0945929839065
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 2508.6146090956972
        NL22:
          techs:
            battery:
              constraints:
                energy_cap_equals: 784.2495152196094
            ccgt:
              constraints:
                energy_cap_equals: 862.4985121538857
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 357.5826593164171
            pv_farm:
              constraints:
                energy_cap_equals: 2500.0
            pv_rooftop:
              constraints:
                energy_cap_equals: 85.8843726587566
            wind_onshore:
              constraints:
                energy_cap_equals: 4942.107396614134
        NL23:
          techs:
            battery:
              constraints:
                energy_cap_equals: 108.53832294784523
            ccgt:
              constraints:
                energy_cap_equals: 150.43812399965162
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 66.18474378705736
            pv_farm:
              constraints:
                energy_cap_equals: 564.3300645728833
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_offshore:
              constraints:
                energy_cap_equals: 91.35565640056272
            wind_onshore:
              constraints:
                energy_cap_equals: 775.5692267976638
        NL31:
          techs:
            battery:
              constraints:
                energy_cap_equals: 861.1912694098621
            ccgt:
              constraints:
                energy_cap_equals: 830.6237294369541
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 308.7125807790655
            pv_farm:
              constraints:
                energy_cap_equals: 2220.4564412444956
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 4870.341949605895
        NL32:
          techs:
            battery:
              constraints:
                energy_cap_equals: 1478.7403244631782
            ccgt:
              constraints:
                energy_cap_equals: 2390.3087724212983
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 500.8107975886412
            pv_farm:
              constraints:
                energy_cap_equals: 2500.0
            pv_rooftop:
              constraints:
                energy_cap_equals: 1405.177971283356
            wind_offshore:
              constraints:
                energy_cap_equals: 874.0725120829129
            wind_onshore:
              constraints:
                energy_cap_equals: 5000.0
        NL33:
          techs:
            battery:
              constraints:
                energy_cap_equals: 1142.1747369056968
            ccgt:
              constraints:
                energy_cap_equals: 2057.682613700397
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 681.6880791012157
            pv_farm:
              constraints:
                energy_cap_equals: 2500.0
            pv_rooftop:
              constraints:
                energy_cap_equals: 4932.076147601722
            wind_offshore:
              constraints:
                energy_cap_equals: 1309.9392776217974
            wind_onshore:
              constraints:
                energy_cap_equals: 5000.0
        NL34:
          techs:
            battery:
              constraints:
                energy_cap_equals: 153.92107052598593
            ccgt:
              constraints:
                energy_cap_equals: 111.62968371553659
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 116.39162448061793
            pv_farm:
              constraints:
                energy_cap_equals: 632.9872133807079
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_offshore:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 947.3849663202808
        NL41:
          techs:
            battery:
              constraints:
                energy_cap_equals: 927.9021632001231
            ccgt:
              constraints:
                energy_cap_equals: 1619.731144796515
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 333.83024925149255
            pv_farm:
              constraints:
                energy_cap_equals: 2500.0
            pv_rooftop:
              constraints:
                energy_cap_equals: 1025.2478227939869
            wind_onshore:
              constraints:
                energy_cap_equals: 5000.0
        NL42:
          techs:
            battery:
              constraints:
                energy_cap_equals: 380.49455334983753
            ccgt:
              constraints:
                energy_cap_equals: 616.4259625391409
            hydrogen_to_power:
              constraints:
                energy_cap_equals: 106.45378696353217
            pv_farm:
              constraints:
                energy_cap_equals: 1309.1187861622911
            pv_rooftop:
              constraints:
                energy_cap_equals: 0.0
            wind_onshore:
              constraints:
                energy_cap_equals: 2557.1843997465667
    low_gas_price:
      techs:
        ccgt:
          costs:
            monetary:
              om_con: 0.05
    res_2h:
      model:
        time:
          function: resample
          function_options:
            resolution: 2H
    res_3h:
      model:
        time:
          function: resample
          function_options:
            resolution: 3H
    res_6h:
      model:
        time:
          function: resample
          function_options:
            resolution: 6H
    weather_2015:
      model:
        subset_time:
        - '2015-01-01'
        - '2015-01-31'
  scenarios:
    2015_weather_low_gas_price:
    - weather_2015
    - low_gas_price
GCOL                                                                                                                                  	               
                                                                                                                                                                                                                                                                             hvac_transmission:NL32                hvac_transmission:NL33                hvac_transmission:NL34                hvac_transmission:NL41                hvac_transmission:NL42                 hydrogen_to_power       !              import_electricity      "       	       lost_load       #              pv_farm $       
       pv_rooftop      %              wind_offshore   &              wind_onshore    '              hvac_transmission:NL12  (              hvac_transmission:NL13  )              hvac_transmission:NL21  *              hvac_transmission:NL22  +              hvac_transmission:NL23  ,              hvac_transmission:NL31  -              export_electricity      .              hvac_transmission       /              hvac_transmission:NL11  0              curtailment     1              demand_electricity      2              ccgt    3              battery 4               5               6              monetary7               8               9               :               ;               <               =               >               ?               @               A               B               C               D              NL21::electricity       E              NL34::electricity       F              NL13::electricity       G              NL31::electricity       H              NL32::electricity       I              NL41::electricity       J              NL23::electricity       K              NL42::electricity       L              NL12::electricity       M              NL33::electricity       N              NL22::electricity       O              NL11::electricity       P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]              NL21::electricity       ^              NL34::electricity       _              NL13::electricity       `              NL31::electricity       a              NL32::electricity       b              NL41::electricity       c              NL23::electricity       d              NL42::electricity       e              NL12::electricity       f              NL33::electricity       g              NL22::electricity       h              NL11::electricity       i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �       )       NL41::hvac_transmission:NL34::electricity       �                       OHDRV         L       L                                 �T     i             �T     i       �     �            ������������������������0        CLASS                DIMENSION_SCALE 6       NAME                 loc_tech_carriers_con 4       _Netcdf4Dimid                            �7�OHDR         v       v                                 ��     L             ��     L       �     Z            ������������������������0        CLASS                DIMENSION_SCALE 7       NAME                 loc_tech_carriers_prod   7��6OHDRJ         �       �                                 ��     �             ��     �     	 ~-     N           ������������������������0        CLASS                DIMENSION_SCALE *       NAME       
          loc_techs 4       _Netcdf4Dimid                            �$%�OHDRd                                                 (                 (           ��     �            ������������������������0        CLASS                DIMENSION_SCALE D       NAME       $          loc_techs_balance_demand_constraint 4       _Netcdf4Dimid                            ����OHDR         �       �                                 �I                  �I            Np     Z            ������������������������0        CLASS                DIMENSION_SCALE /       NAME                 loc_techs_cost   �a1                              [��FRHP               ���������.      �k      @                    h                                                         q�      ����FSHD  k                             P x          ��     H       H       %���BTLF �?�    wm- ]  " �8 8  ' �!2 ]
   �P� {  + oK	 �	   t�	 �
   �2�   ! �B� �  - �=] �  $ 1�� �  6 vv� 6  1 ~�W �	    ͯ� �  ! +˾ �   ( ���  �  # �s�# �   \mK& O  $ ٽ�* 9  + �u�. �  % aL/ g  " ڞu/    �a�/ 
  + »�2 �   ) ��9 A  7 @MNI 
  6 ³�L �  " )m�M �  & y��P -    o�6Q q	  ) ��-S B  , ��S   ) �`T n    � V �  ' 6�gV �   &�V @  ! F�Y    .��Z �
   ��-                                                                                    BTLF              8        -    9        E    :        d   9 ;        �   ( <        �   ) =        �    >        
  6 ?        @  ! @        a  7 A        �  , B        �  $ C          " D        7   E        @
   F        ]
   G        |
   H        �   ���9                                                                                                                                                                                                                                        OCHK    1     �       4       _Netcdf4Dimid                            *�R�OHDRe         j       j                                 b     g             b     g       �     �            ������������������������0        CLASS                DIMENSION_SCALE E       NAME       %          loc_techs_cost_investment_constraint 4       _Netcdf4Dimid                       	     h[�VOHDRZ         j       j                                 �x     �             �x     �      ��                ������������������������0        CLASS                DIMENSION_SCALE :       NAME                 loc_techs_investment_cost 4       _Netcdf4Dimid                       
     �;HyOHDRR         f       f                                 _�     T             _�     T      xS                ������������������������0        CLASS                DIMENSION_SCALE 2       NAME                 loc_techs_om_cost 4       _Netcdf4Dimid                            YWcOHDRP                                                 ��     �             ��     �      QY                ������������������������0        CLASS                DIMENSION_SCALE 0       NAME                 loc_techs_store 4       _Netcdf4Dimid                            0���OHDRw         �       �                 @     ��     _                   �    �     �           ������������������������0        CLASS                DIMENSION_SCALE *       NAME       
          timesteps 4       _Netcdf4Dimid                          ?       units                hours since 2019-01-01 00:00:00V�>OCHK            �4       _Netcdf4Dimid                          A�w�OCHK            �4       _Netcdf4Dimid                          ��OCHK            �4       _Netcdf4Dimid                          9��UOCHK            �4       _Netcdf4Dimid                          9��UOHDRN                                                 ��                 ��           `�                  ������������������������0        CLASS                DIMENSION_SCALE .       NAME                 carrier_tiers 4       _Netcdf4Dimid                          P�OHDRL                                                 ��                 ��           ��     �            ������������������������0        CLASS                DIMENSION_SCALE ,       NAME                 coordinates 4       _Netcdf4Dimid                            ���?OHDRu         L       L                                 ��                 ��           f�
     �           ������������������������0        CLASS                DIMENSION_SCALE U       NAME       5          loc_tech_carriers_carrier_consumption_max_constraint 4       _Netcdf4Dimid                          dJx:OHDR:         v       v                                 &�
     ]             &�
     ]       ��
     Z            ������������������������0        CLASS                DIMENSION_SCALE T       NAME       4          loc_tech_carriers_carrier_production_max_constraint   �}�SOHDRY                                                 �     V             �     V       �<     �           ������������������������0        CLASS                DIMENSION_SCALE 9       NAME                 loc_tech_carriers_demand 4       _Netcdf4Dimid                          A(��OHDRe                                                 �     �             �     �       >     �            ������������������������0        CLASS                DIMENSION_SCALE E       NAME       %          loc_tech_carriers_ramping_constraint 4       _Netcdf4Dimid                          ���OHDR]         B       B                                 �     �             �     �       �>                 ������������������������0        CLASS                DIMENSION_SCALE =       NAME                 loc_tech_carriers_supply_all 4       _Netcdf4Dimid                          :�1OHDRh         B       B                                 �B     <             �B     <       �R                 ������������������������0        CLASS                DIMENSION_SCALE H       NAME       (          loc_tech_carriers_supply_conversion_all 4       _Netcdf4Dimid                          �ִOHDR+                                                 W     =             W     =       =�
     Z            ������������������������0        CLASS                DIMENSION_SCALE E       NAME       %          loc_techs_balance_storage_constraint   �R�OHDRd         6       6                                 �     �             �     �       �h     `           ������������������������0        CLASS                DIMENSION_SCALE D       NAME       $          loc_techs_balance_supply_constraint 4       _Netcdf4Dimid                          i��OHDRj                                                 �k     k             �k     k       �{     �           ������������������������0        CLASS                DIMENSION_SCALE J       NAME       *          loc_techs_balance_transmission_constraint 4       _Netcdf4Dimid                          v!W�OHDRZ         �       �                                 �}     
             �}     
       ��                 ������������������������0        CLASS                DIMENSION_SCALE :       NAME                 loc_techs_cost_constraint 4       _Netcdf4Dimid                          ף��OHDR^         f       f                                 ƕ     [             ƕ     [       ƥ     `           ������������������������0        CLASS                DIMENSION_SCALE >       NAME                 loc_techs_cost_var_constraint 4       _Netcdf4Dimid                          ��o�OHDRQ                                                 &�     �             &�     �       &�     �           ������������������������0        CLASS                DIMENSION_SCALE 1       NAME                 loc_techs_demand 4       _Netcdf4Dimid                          ;uۣOHDR+         �       �                                 &�     �             &�     �       &�     Z            ������������������������0        CLASS                DIMENSION_SCALE E       NAME       %          loc_techs_energy_capacity_constraint   \��gOHDRq                                                 `�     �             `�     �       ��     �            ������������������������0        CLASS                DIMENSION_SCALE Q       NAME       1          loc_techs_energy_capacity_storage_max_constraint 4       _Netcdf4Dimid                          �~!�OHDRZ         N       N                                 `�     �             `�     �      �R     �            ������������������������0        CLASS                DIMENSION_SCALE :       NAME                 loc_techs_finite_resource 4       _Netcdf4Dimid                            ��EOHDRa                                                 `�     G            `�     G      `     �           ������������������������0        CLASS                DIMENSION_SCALE A       NAME       !          loc_techs_finite_resource_demand 4       _Netcdf4Dimid                           �!�4OHDR          6       6                                 `�     x            `�     x      �     `           ������������������������0        CLASS                DIMENSION_SCALE     2�     �                �P>T         �&�9BTIN I        �
   k        �  - �           Nn     2�     !PP     �V     L�{                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF J        �
   K        �
   L        �
  " M           N        .  ! O        O  $ P        s  & Q        �   R        �  " S        S   T          - U        B  , V        n  ' W        n    X        �   Y        �  G Z        �  F [        9  + \        d  7 ]        �  / ^          : _        O  7 `        �  6 a        �  < b        �  , c        �  0 d          # e        A  7 f        x  C g        �  , h          3 i        H  3 j        {  + ��       OCHKA       NAME       !          loc_techs_finite_resource_supply 4       _Netcdf4Dimid                       !   ��SOHDRY         �       �                                 `�     �            `�     �      N�     �            ������������������������0        CLASS                DIMENSION_SCALE 9       NAME                 loc_techs_non_conversion 4       _Netcdf4Dimid                       "     9�OHDR[         r       r                                  +     �              +     �        K                 ������������������������0        CLASS                DIMENSION_SCALE ;       NAME                 loc_techs_non_transmission 4       _Netcdf4Dimid                       #   �OHDRY         B       B                                 @R     _             @R     _       @b                 ������������������������0        CLASS                DIMENSION_SCALE 9       NAME                 loc_techs_om_cost_supply 4       _Netcdf4Dimid                       $   3	��OHDRR                                                 `f     X             `f     X       F�     �            ������������������������0        CLASS                DIMENSION_SCALE 2       NAME                 loc_techs_ramping 4       _Netcdf4Dimid                       %   z��OHDRR                                                 `f     q             `f     q       `v     �           ������������������������0        CLASS                DIMENSION_SCALE 2       NAME                 loc_techs_storage 4       _Netcdf4Dimid                       &   Q�yOHDRf                                                 �w                  �w            ��     �           ������������������������0        CLASS                DIMENSION_SCALE F       NAME       &          loc_techs_storage_capacity_constraint 4       _Netcdf4Dimid                       '   ��="OHDR+                                                 �w     =             �w     =       ��     Z            ������������������������0        CLASS                DIMENSION_SCALE E       NAME       %          loc_techs_storage_initial_constraint   5��OHDRa                                                 �w     n             �w     n       ��     �           ������������������������0        CLASS                DIMENSION_SCALE A       NAME       !          loc_techs_storage_max_constraint 4       _Netcdf4Dimid                       )   �W��OHDRQ         B       B                                 :�                  :�            :�                 ������������������������0        CLASS                DIMENSION_SCALE 1       NAME                 loc_techs_supply 4       _Netcdf4Dimid                       *   ����OHDRU         B       B                                 :�     �             :�     �       Z�                 ������������������������0        CLASS                DIMENSION_SCALE 5       NAME                 loc_techs_supply_all 4       _Netcdf4Dimid                       +   ���OHDR`         B       B                                 z�     ]             z�     ]       z�                 ������������������������0        CLASS                DIMENSION_SCALE @       NAME                  loc_techs_supply_conversion_all 4       _Netcdf4Dimid                       ,   ?���OHDR                                                  ��     :             ��     :       `�     �           ������������������������0        CLASS                DIMENSION_SCALE     �     �                �                                    �B�BTIN ��] O  7 $��� �  3 Nl     &�     &g�     "��P                                                                                                                                                                                                                                                                                                                                                                                                                                                                     BTLF ���] ~  7 �d �  + O�mi   # FY*j |
   ,{n H  3 o=�n �   �Elo F  8 ̹�p �
  " {�t �  0 ��{ �    F��| �  / �T>}   " �$�� 3	  > O�� �   �lo� �  ) 0d�� �  F M߫� �   �<� �   T��� �   1M7� �  " n�� �    U�� �  > �X� "  $ �N� �	   �(�� x  C �9p� �
   �ԕ� i   %��   : �Fݵ _  2 ��_� 7   J鱷 n  ' �Pr� 	   �t1� �  , �� a  7 �a�� s  & 7���   - ���� �  5 Nr� �  , %-~                                                                                    OCHKL       NAME       ,          loc_techs_symmetric_transmission_constraint 4       _Netcdf4Dimid                       -   �X|OHDRW                                                 ��     s             ��     s      ��     �            ������������������������0        CLASS                DIMENSION_SCALE 7       NAME                 loc_techs_transmission 4       _Netcdf4Dimid                       .     �Jl�OHDRE                                                 ��     �             ��     �       ��     �            ������������������������0        CLASS                DIMENSION_SCALE %       NAME                 locs 4       _Netcdf4Dimid                       /     �zmOHDRJ                                                 ��     �             ��     �       �                  ������������������������0        CLASS                DIMENSION_SCALE *       NAME       
          resources 4       _Netcdf4Dimid                       0   �@aOHDRM                                                 ��     �             ��     �       Z�     0            ������������������������0        CLASS                DIMENSION_SCALE -       NAME                 techs_demand 4       _Netcdf4Dimid                       1   �΀7OHDRl                                                 ��     �             ��     �       ��                 ������������������������0        CLASS                DIMENSION_SCALE L       NAME       ,          techs_energy_capacity_systemwide_constraint 4       _Netcdf4Dimid                       2   �+jIOHDRW                                                 ��     �             ��     �       ��     �            ������������������������0        CLASS                DIMENSION_SCALE 7       NAME                 techs_non_transmission 4       _Netcdf4Dimid                       3   ��uOHDRN                                                 ��     �             ��     �       J�                  ������������������������0        CLASS                DIMENSION_SCALE .       NAME                 techs_storage 4       _Netcdf4Dimid                       4   ��a�OHDRM                                                 ��     �             ��     �       j�     p            ������������������������0        CLASS                DIMENSION_SCALE -       NAME                 techs_supply 4       _Netcdf4Dimid                       5   ���OHDRS                                                 ��                  ��            ��     �            ������������������������0        CLASS                DIMENSION_SCALE 3       NAME                 techs_transmission 4       _Netcdf4Dimid                       6   �l��OHDRY                                                 ��                 ��           ��                 ������������������������0        CLASS                DIMENSION_SCALE 9       NAME                 techs_transmission_names 4       _Netcdf4Dimid                       7   �x��  _Netcdf4Dimid                       7   �v�hFSSE 1      [  �    x �    �             
 A �J    ���FSSE �k       �     �	     �     �   f �
   �2�XFHDB jk      
  ��       techs_transmission_names �     E       
energy_cap�     F       carrier_prod�     G       carrier_conV�     I       storage_cap�1     J       storage�4     K       cost_var�"     L       cost_investmentP%     M       unmet_demand�'     N       system_balance,�     O       required_resource�7     P       cost_investment_rhsc�     Q       cost_var_rhs�T     R       capacity_factorm�     ~       techs_transmission��      FHDB jk        V~[+r       loc_techs_supply��     s       loc_techs_supply_all�     t       loc_techs_supply_conversion_all|�     u       +loc_techs_symmetric_transmission_constraint�     v       loc_techs_transmission��     x       	resourcesd�     y       techs_demand��     z       +techs_energy_capacity_systemwide_constraint�     {       techs_non_transmission��     |       techs_storage�     }       techs_supplyH�     �       resource��     onԷ       FHDB jk        �4��h        loc_techs_finite_resource_demand��     i        loc_techs_finite_resource_supply�     j       loc_techs_non_conversion��     k       loc_techs_non_transmission �     l       loc_techs_om_cost_supply��     m       loc_techs_ramping�     n       loc_techs_storageJ�     o       %loc_techs_storage_capacity_constraint��     p       $loc_techs_storage_initial_constraint�     q        loc_techs_storage_max_constraintQ�     H       costy      FHDB jk        �b�-^       'loc_tech_carriers_supply_conversion_allԂ     _       $loc_techs_balance_storage_constraintH�     a       )loc_techs_balance_transmission_constraint�     b       loc_techs_cost_constrainte�     c       loc_techs_cost_var_constraintˉ     d       loc_techs_demand5�     e       $loc_techs_energy_capacity_constraint��     f       0loc_techs_energy_capacity_storage_max_constraintɍ     g       loc_techs_finite_resourceF�     �       colors��     FHDB jk        ˢ�C       loc_techs_store�t     D       	timesteps9v     S       carriersY�     W       carrier_tiersy     X       coordinatesvz     Y       4loc_tech_carriers_carrier_consumption_max_constraint�{     Z       3loc_tech_carriers_carrier_production_max_constraintO}     [       loc_tech_carriers_demand�~     \       $loc_tech_carriers_ramping_constraint�     ]       loc_tech_carriers_supply_allk�     `       #loc_techs_balance_supply_constraint�     FHDB jk         t.D�        techs��     8       costs�     9       loc_carriersB�     :       &loc_carriers_system_balance_constraint��     ;       loc_tech_carriers_con�d     <       loc_tech_carriers_prod8f     =       	loc_techsag     >       #loc_techs_balance_demand_constraint�h     ?       loc_techs_cost'j     @       $loc_techs_cost_investment_constraint�p     A       loc_techs_investment_costr     B       loc_techs_om_costs     w       locs�      FHIB jk         �     �     �     �     �     �     ��
     �S     ��v`FHDB S          ;��,    run_config         J         backend: pyomo
bigM: 1000000.0
cyclic_storage: true
ensure_feasibility: true
mode: plan
objective_options:
  cost_class:
    monetary: 1
  sense: minimize
objective: minmax_cost_optimization
operation:
  window:
  horizon:
  use_cap_results: false
spores_options:
  spores_number: 3
  slack: 0.1
  score_cost_class: spores_score
  objective_cost_class:
    spores_score: 1
    monetary: 0
  slack_cost_group:
  save_per_spore: false
  save_per_spore_path:
  skip_cost_op: false
relax_constraint:
  demand_share_per_timestep_decision_main_constraint: 0
solver: cbc
zero_threshold: 1e-10
                                                                                                                                                                                                                                                                                                                                                                                        FHDB S       @   0�B� 	   defaults           �
         carrier_ratios:
charge_rate:
energy_cap_per_storage_cap_min: 0
energy_cap_per_storage_cap_max: .inf
energy_cap_per_storage_cap_equals:
energy_cap_equals:
energy_cap_equals_systemwide:
energy_cap_max: .inf
energy_cap_max_systemwide: .inf
energy_cap_min: 0
energy_cap_min_use: 0
energy_cap_per_unit:
energy_cap_scale: 1.0
energy_con: false
energy_eff: 1.0
energy_eff_per_distance: 1.0
energy_prod: false
energy_ramping:
export_cap: .inf
export_carrier:
force_asynchronous_prod_con: false
force_resource: false
lifetime:
one_way: false
parasitic_eff: 1.0
resource: 0
resource_area_equals:
resource_area_max: .inf
resource_area_min: 0
resource_area_per_energy_cap:
resource_cap_equals:
resource_cap_equals_energy_cap: false
resource_cap_max: .inf
resource_cap_min: 0
resource_eff: 1.0
resource_min_use: 0
resource_scale: 1.0
resource_unit: energy
storage_cap_equals:
storage_cap_max: .inf
storage_cap_min: 0
storage_cap_per_unit:
storage_discharge_depth: 0
storage_initial: 0
storage_loss: 0
units_equals:
units_equals_systemwide:
units_max:
units_max_systemwide: .inf
units_min:
cost_energy_cap: 0
cost_energy_cap_per_distance: 0
cost_export: 0
cost_interest_rate: 0
cost_om_annual: 0
cost_om_annual_investment_fraction: 0
cost_om_con: 0
cost_om_prod: 0
cost_purchase: 0
cost_resource_area: 0
cost_resource_cap: 0
cost_storage_cap: 0
cost_depreciation_rate: 1
transmission_node: false
coordinates: {}
available_area:
techs: None
exists: true
group_demand_share_min: 0
group_demand_share_max: .inf
group_demand_share_equals:
group_demand_share_per_timestep_min: 0
group_demand_share_per_timestep_max: .inf
group_demand_share_per_timestep_equals:
group_demand_share_per_timestep_decision:
group_carrier_prod_share_min: 0
group_carrier_prod_share_max: .inf
group_carrier_prod_share_equals:
group_carrier_prod_share_per_timestep_min: 0
group_carrier_prod_share_per_timestep_max: .inf
group_carrier_prod_share_per_timestep_equals:
group_net_import_share_min:
group_net_import_share_max:
group_net_import_share_equals:
group_carrier_prod_min:
group_carrier_prod_max:
group_carrier_prod_equals:
group_carrier_con_min:
group_carrier_con_max:
group_carrier_con_equals:
group_cost_min:
group_cost_max:
group_cost_equals:
group_cost_var_min:
group_cost_var_max:
group_cost_var_equals:
group_cost_investment_min:
group_cost_investment_max:
group_cost_investment_equals:
group_energy_cap_share_min: 0
group_energy_cap_share_max: .inf
group_energy_cap_share_equals:
group_energy_cap_min: 0
group_energy_cap_max: .inf
group_energy_cap_equals:
group_storage_cap_min: 0
group_storage_cap_max: .inf
group_storage_cap_equals:
group_resource_area_min: 0
group_resource_area_max: .inf
group_resource_area_equals:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 FHDB S           ��    termination_condition                optimal    objective_function_value         ?      @ 4 4�                         x���eJA    solution_time    ?      @ 4 4�                         �3�%�B@    time_finished                2024-11-25 14:43:32    calliope_version                     0.6.10    applied_overrides                    res_3h 	   scenario                     res_3h    allow_operate_mode             @                                   model_config                calliope_version: 0.6.10
name: Dutch power system at NUTS2 resolution
subset_time:
- '2019-01-01'
- '2019-01-31'
time:
  function: resample
  function_options:
    resolution: 3H
timeseries_data_path: model_files/timeseries_data
timeseries_dateformat: '%Y-%m-%d %H:%M:%S'
    _NCProperties      #          version=2,netcdf=4.7.4,hdf5=1.12.0,                                                                                           FHIB S           o�     o�     ��������������������������������������������������������������������������������o�     ������������������������M�?B   �T     3      �T     2      �T     0      �T     1      �T     -      �T     .      �T     /      �T     '      �T     (      �T     )      �T     *      �T     +      �T     ,      �T           �T           �T           �T           �T           �T            �T     !   	   �T     "      �T     #   
   �T     $      �T     %      �T     &      �T     6      �T     O      �T     N      �T     M      �T     J      �T     K      �T     L      �T     D      �T     E      �T     F      �T     G      �T     H      �T     I      �T     h      �T     g      �T     f      �T     c      �T     d      �T     e      �T     ]      �T     ^      �T     _      �T     `      �T     a      �T     b   )   ��     K   )   ��     J   )   ��     I      ��     G      ��     H   )   ��     B   )   ��     C   )   ��     D      ��     E      ��     F   $   ��     9      ��     :   %   ��     ;   %   ��     <      ��     =      ��     >   $   ��     ?   )   ��     @      ��     A   $   ��     &      ��     '   %   ��     (   %   ��     )   )   ��     *   )   ��     +   )   ��     ,   )   ��     -      ��     .   $   ��     /   $   ��     0   )   ��     1   $   ��     2   %   ��     3   )   ��     4   $   ��     5   )   ��     6   )   ��     7   )   ��     8   )   �T     �   %   ��        %   ��        )   ��           ��        $   ��        )   ��           ��        )   ��        %   ��     	   $   ��     
   )   ��           ��        $   ��           ��           ��           ��        %   ��           ��        )   ��        %   ��           ��           ��        %   ��        $   ��           ��        )   ��        )   ��        )   ��        )   ��           ��           ��        )   ��         $   ��     !   )   ��     "      ��     #   %   ��     $      ��     %   GCOL                  %       NL21::demand_electricity::electricity          %       NL22::demand_electricity::electricity          )       NL21::hvac_transmission:NL22::electricity                     NL23::curtailment::electricity         $       NL41::hydrogen_to_power::electricity           )       NL32::hvac_transmission:NL33::electricity                     NL13::curtailment::electricity         )       NL23::hvac_transmission:NL21::electricity       	       %       NL12::demand_electricity::electricity   
       $       NL23::hydrogen_to_power::electricity           )       NL22::hvac_transmission:NL21::electricity                     NL31::battery::electricity             $       NL22::hydrogen_to_power::electricity                  NL31::curtailment::electricity                NL41::battery::electricity                    NL23::battery::electricity             %       NL13::demand_electricity::electricity                 NL12::battery::electricity             )       NL33::hvac_transmission:NL41::electricity              %       NL23::demand_electricity::electricity                 NL34::curtailment::electricity                NL41::curtailment::electricity         %       NL31::demand_electricity::electricity          $       NL11::hydrogen_to_power::electricity                  NL12::curtailment::electricity         )       NL31::hvac_transmission:NL32::electricity              )       NL12::hvac_transmission:NL23::electricity              )       NL22::hvac_transmission:NL41::electricity              )       NL21::hvac_transmission:NL13::electricity                     NL32::curtailment::electricity                NL11::battery::electricity              )       NL23::hvac_transmission:NL12::electricity       !       $       NL12::hydrogen_to_power::electricity    "       )       NL13::hvac_transmission:NL21::electricity       #              NL42::battery::electricity      $       %       NL41::demand_electricity::electricity   %              NL22::curtailment::electricity  &       $       NL21::hydrogen_to_power::electricity    '              NL13::battery::electricity      (       %       NL32::demand_electricity::electricity   )       %       NL42::demand_electricity::electricity   *       )       NL21::hvac_transmission:NL23::electricity       +       )       NL41::hvac_transmission:NL33::electricity       ,       )       NL33::hvac_transmission:NL31::electricity       -       )       NL11::hvac_transmission:NL13::electricity       .              NL32::battery::electricity      /       $       NL31::hydrogen_to_power::electricity    0       $       NL13::hydrogen_to_power::electricity    1       )       NL34::hvac_transmission:NL41::electricity       2       $       NL33::hydrogen_to_power::electricity    3       %       NL33::demand_electricity::electricity   4       )       NL42::hvac_transmission:NL41::electricity       5       $       NL32::hydrogen_to_power::electricity    6       )       NL11::hvac_transmission:NL12::electricity       7       )       NL12::hvac_transmission:NL11::electricity       8       )       NL33::hvac_transmission:NL32::electricity       9       $       NL42::hydrogen_to_power::electricity    :              NL22::battery::electricity      ;       %       NL34::demand_electricity::electricity   <       %       NL11::demand_electricity::electricity   =              NL21::curtailment::electricity  >              NL42::curtailment::electricity  ?       $       NL34::hydrogen_to_power::electricity    @       )       NL31::hvac_transmission:NL33::electricity       A              NL34::battery::electricity      B       )       NL32::hvac_transmission:NL23::electricity       C       )       NL23::hvac_transmission:NL32::electricity       D       )       NL13::hvac_transmission:NL11::electricity       E              NL33::curtailment::electricity  F              NL11::curtailment::electricity  G              NL33::battery::electricity      H              NL21::battery::electricity      I       )       NL32::hvac_transmission:NL31::electricity       J       )       NL41::hvac_transmission:NL22::electricity       K       )       NL41::hvac_transmission:NL42::electricity       L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               NL11::wind_offshore::electricity�       $       NL31::hydrogen_to_power::electricity    �              NL41::lost_load::electricity    �              NL23::pv_farm::electricity      �              NL11::lost_load::electricity    �       $       NL33::hydrogen_to_power::electricity    �              NL31::pv_rooftop::electricity   �       )       NL33::hvac_transmission:NL32::electricity       �       )       NL32::hvac_transmission:NL33::electricity       �       $       NL22::hydrogen_to_power::electricity    �              NL22::lost_load::electricity    �              NL32::lost_load::electricity    �       )       NL33::hvac_transmission:NL41::electricity       �              NL22::wind_onshore::electricity �              NL41::pv_rooftop::electricity   �              NL33::wind_onshore::electricity �              NL12::pv_rooftop::electricity   �              NL33::pv_rooftop::electricity   �              NL11::pv_farm::electricity      �              NL23::pv_rooftop::electricity   �              NL23::ccgt::electricity �              NL11::battery::electricity      �              NL32::pv_farm::electricity      �       )       NL41::hvac_transmission:NL22::electricity       �              NL34::wind_onshore::electricity �       )       NL32::hvac_transmission:NL23::electricity       �       )       NL23::hvac_transmission:NL32::electricity       �              NL21::ccgt::electricity �              NL13::ccgt::electricity �       )       NL31::hvac_transmission:NL33::electricity       �              NL34::battery::electricity      �               NL23::wind_offshore::electricity�              NL34::lost_load::electricity    �       )       NL21::hvac_transmission:NL23::electricity       �              NL11::ccgt::electricity �              NL31::pv_farm::electricity      �       $       NL13::hydrogen_to_power::electricity    �       )       NL34::hvac_transmission:NL41::electricity       �       )       NL42::hvac_transmission:NL41::electricity       �               OCHK   '�     �       4       _Netcdf4Dimid                            ���G   �     O      �     N   $   �     L      �     M      �     H   $   �     I   )   �     J      �     K      �     A   )   �     B      �     C      �     D      �     E      �     F       �     G   )   �     2   )   �     3   )   �     4      �     5   $   �     6   $   �     7   )   �     8   )   �     9       �     :   )   �     ;   )   �     <      �     =      �     >      �     ?      �     @   $   �           �           �           �        )   �        )   �            �        )   �        $   �           �           �           �            �     !      �     "      �     #       �     $      �     %      �     &      �     '      �     (      �     )      �     *      �     +      �     ,      �     -      �     .   )   �     /   )   �     0      �     1       ��     �   $   ��     �      ��     �      ��     �      ��     �   $   ��     �      ��     �   )   ��     �   )   ��     �   $   ��     �      ��     �      ��     �   )   ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   )   ��     �      ��     �   )   ��     �   )   ��     �      ��     �      ��     �   )   ��     �      ��     �       ��     �      ��     �   )   ��     �      ��     �      ��     �   $   ��     �   )   ��     �   )   ��     �      �        )   �           �           �           �           �        $   �        )   �           �     	      �     
      �           �           �           �        $   �           �           �        )   �        )   �           �        GCOL                        NL42::lost_load::electricity           )       NL12::hvac_transmission:NL11::electricity                     NL13::wind_onshore::electricity               NL34::ccgt::electricity               NL12::lost_load::electricity                  NL13::battery::electricity             $       NL23::hydrogen_to_power::electricity           )       NL23::hvac_transmission:NL21::electricity       	              NL31::lost_load::electricity    
              NL21::pv_rooftop::electricity                 NL13::pv_farm::electricity                    NL23::wind_onshore::electricity               NL32::pv_rooftop::electricity                 NL32::wind_onshore::electricity        $       NL11::hydrogen_to_power::electricity                  NL33::pv_farm::electricity                    NL42::pv_farm::electricity             )       NL22::hvac_transmission:NL41::electricity              )       NL23::hvac_transmission:NL12::electricity                     NL42::battery::electricity             $       NL34::hydrogen_to_power::electricity                  NL21::lost_load::electricity                  NL32::battery::electricity                    NL34::pv_farm::electricity             )       NL11::hvac_transmission:NL12::electricity              )       NL41::hvac_transmission:NL34::electricity                      NL33::wind_offshore::electricity       )       NL21::hvac_transmission:NL22::electricity              $       NL41::hydrogen_to_power::electricity                  NL13::lost_load::electricity                  NL33::lost_load::electricity                   NL42::ccgt::electricity !              NL12::wind_onshore::electricity "              NL21::pv_farm::electricity      #              NL41::ccgt::electricity $               NL12::wind_offshore::electricity%              NL11::wind_onshore::electricity &              NL31::battery::electricity      '              NL11::pv_rooftop::electricity   (              NL23::lost_load::electricity    )              NL31::ccgt::electricity *              NL41::pv_farm::electricity      +              NL42::wind_onshore::electricity ,              NL33::ccgt::electricity -              NL31::wind_onshore::electricity .              NL22::ccgt::electricity /       )       NL33::hvac_transmission:NL31::electricity       0       )       NL11::hvac_transmission:NL13::electricity       1              NL22::pv_rooftop::electricity   2       )       NL31::hvac_transmission:NL32::electricity       3       )       NL12::hvac_transmission:NL23::electricity       4       )       NL21::hvac_transmission:NL13::electricity       5              NL12::pv_farm::electricity      6       $       NL12::hydrogen_to_power::electricity    7       $       NL32::hydrogen_to_power::electricity    8       )       NL13::hvac_transmission:NL11::electricity       9       )       NL13::hvac_transmission:NL21::electricity       :               NL34::wind_offshore::electricity;       )       NL41::hvac_transmission:NL42::electricity       <       )       NL32::hvac_transmission:NL31::electricity       =              NL34::pv_rooftop::electricity   >              NL21::battery::electricity      ?              NL41::wind_onshore::electricity @              NL12::ccgt::electricity A              NL13::pv_rooftop::electricity   B       )       NL22::hvac_transmission:NL21::electricity       C              NL21::wind_onshore::electricity D              NL41::battery::electricity      E              NL23::battery::electricity      F              NL12::battery::electricity      G               NL32::wind_offshore::electricityH              NL42::pv_rooftop::electricity   I       $       NL21::hydrogen_to_power::electricity    J       )       NL41::hvac_transmission:NL33::electricity       K              NL32::ccgt::electricity L       $       NL42::hydrogen_to_power::electricity    M              NL22::battery::electricity      N              NL22::pv_farm::electricity      O              NL33::battery::electricity         (          (       
   (          (          (          (          (          (          (          (     
     (          (          (          (          (          (          (          (          (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (           (          (          (       
   (          (          (          (          (       
   (     	  
   (     �      (     �   
   (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �   
   (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �      (     �   GCOL                                                                                                                                   	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL13::hvac_transmission:NL21    �              NL21::demand_electricity�              NL32::wind_offshore     �              NL42::demand_electricity�              NL12::pv_rooftop�              NL32::hydrogen_to_power �              NL34::wind_onshore      �              NL33::demand_electricity�              NL21::pv_rooftop�              NL13::wind_onshore      �       
       NL13::ccgt      �              NL23::hvac_transmission:NL21    �              NL12::demand_electricity�              NL12::hydrogen_to_power �              NL34::pv_rooftop�              NL23::hvac_transmission:NL12    �              NL33::hvac_transmission:NL31    �              NL34::hvac_transmission:NL41    �              NL23::hydrogen_to_power �              NL21::battery   �              NL21::curtailment       �              NL11::hvac_transmission:NL13    �              NL13::pv_rooftop�              NL41::wind_onshore      �       
       NL22::ccgt      �              NL11::lost_load �              NL11::pv_farm   �              NL32::lost_load �              NL33::pv_rooftop�              NL22::hvac_transmission:NL21    �              NL32::pv_rooftop�              NL34::hydrogen_to_power �              NL21::wind_onshore      �              NL21::hvac_transmission:NL23    �              NL34::demand_electricity�              NL11::battery   �              NL33::pv_farm   �              NL33::hydrogen_to_power �              NL34::lost_load �              NL33::curtailment       �              NL42::lost_load �              NL22::curtailment       �              NL22::hvac_transmission:NL41    �              NL32::hvac_transmission:NL31    �              NL42::pv_farm   �              NL31::curtailment       �              NL31::lost_load �              NL33::hvac_transmission:NL32    �              NL11::wind_onshore      �              NL21::pv_farm   �              NL32::curtailment       �       
       NL11::ccgt      �              NL34::wind_offshore     �              NL31::demand_electricity�              NL33::wind_onshore      �              NL31::battery   �              NL12::lost_load �              NL31::pv_rooftop�       
       NL34::ccgt      �              NL22::pv_rooftop�              NL41::hvac_transmission:NL33    �              NL22::hydrogen_to_power �              NL11::curtailment       �              NL11::pv_rooftop�              NL13::battery   �              NL31::hvac_transmission:NL33    �              NL12::hvac_transmission:NL23    �              NL23::pv_farm   �              NL31::pv_farm   �              NL42::wind_onshore      �              NL34::pv_farm   �       
       NL42::ccgt      �              NL41::hvac_transmission:NL34    �       
       NL12::ccgt      �              NL22::pv_farm   �       
       NL32::ccgt      �              NL21::hydrogen_to_power �              NL32::hvac_transmission:NL33    �              NL23::curtailment       �              NL11::wind_offshore     �              NL32::hvac_transmission:NL23    �              NL33::hvac_transmission:NL41    �              NL41::demand_electricity�              NL22::demand_electricity�              NL13::pv_farm   �              NL33::lost_load �       
       NL33::ccgt      �              NL41::battery   �              NL13::hydrogen_to_power �              NL42::battery   �              NL33::wind_offshore     �              NL41::lost_load �              NL13::hvac_transmission:NL11    �              NL12::hvac_transmission:NL11    �              NL32::demand_electricity�              NL32::wind_onshore      �              NL12::curtailment       �              NL21::lost_load �              NL41::hydrogen_to_power �              NL23::wind_offshore     �              NL41::hvac_transmission:NL22    �              NL13::curtailment       �              NL42::pv_rooftop�              NL31::hvac_transmission:NL32    �              NL22::lost_load �              NL33::battery   �              NL41::curtailment       �       
       NL41::ccgt      �              NL12::wind_onshore      �              NL42::curtailment       �              NL12::wind_offshore     �              NL12::pv_farm   �              NL23::battery   �              NL41::hvac_transmission:NL42                  NL23::demand_electricity             NL22::wind_onshore                   NL21::hvac_transmission:NL13                 NL32::pv_farm         
       NL31::ccgt                   NL23::lost_load              NL31::hydrogen_to_power              NL13::lost_load              NL22::battery   	      
       NL23::ccgt      
             NL13::demand_electricity             NL11::demand_electricity             NL34::battery                NL34::curtailment                    NL42::hydrogen_to_power              NL11::hvac_transmission:NL12                 NL23::hvac_transmission:NL32                 NL42::hvac_transmission:NL41                 NL21::hvac_transmission:NL22                 NL31::wind_onshore                   NL41::pv_rooftop             NL32::battery                NL23::pv_rooftop             NL23::wind_onshore                   NL12::battery         
       NL21::ccgt                   NL11::hydrogen_to_power              NL41::pv_farm                                                                          !              "              #              $              %              &              '              (              )              *              +              ,              -              .              /              0              1              2              3              4              5             NL33::curtailment       6             NL22::curtailment       7                 �I           �I           �I           �I           �I           �I           �I           �I           �I           �I           �I           �I           (     5     (     6     �I           �I           �I           �I           �I           �I           �I           �I           �I     	      �I     
   GCOL                        NL41::demand_electricity              NL22::demand_electricity              NL31::curtailment                     NL32::curtailment                     NL31::demand_electricity              NL32::demand_electricity              NL12::curtailment                     NL11::curtailment       	              NL13::curtailment       
              NL23::curtailment                     NL34::curtailment                     NL21::curtailment                     NL41::curtailment                     NL42::curtailment                     NL23::demand_electricity              NL34::demand_electricity              NL13::demand_electricity              NL11::demand_electricity              NL12::demand_electricity              NL33::demand_electricity              NL42::demand_electricity              NL21::demand_electricity                                                                                                                                                       !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL32::wind_offshore     �              NL12::pv_rooftop�              NL32::hydrogen_to_power �              NL34::wind_onshore      �              NL21::pv_rooftop�              NL13::wind_onshore      �       
       NL13::ccgt      �              NL23::hvac_transmission:NL21    �              NL12::hydrogen_to_power �              NL34::pv_rooftop�              NL33::hvac_transmission:NL31    �              NL23::hvac_transmission:NL12    �              NL34::hvac_transmission:NL41    �              NL23::hydrogen_to_power �              NL21::battery   �              NL21::curtailment       �              NL11::hvac_transmission:NL13    �              NL13::pv_rooftop�              NL41::wind_onshore      �       
       NL22::ccgt      �              NL11::lost_load �              NL11::pv_farm   �              NL32::lost_load �              NL33::pv_rooftop�              NL22::hvac_transmission:NL21    �              NL32::pv_rooftop�              NL34::hydrogen_to_power �              NL21::wind_onshore      �               �                  b     f      b     e   
   b     d      b     b      b     c      b     ^      b     _      b     `      b     a      b     V      b     W      b     X      b     Y      b     Z      b     [      b     \      b     ]   
   b     F      b     G      b     H      b     I      b     J      b     K      b     L      b     M      b     N      b     O   
   b     P      b     Q      b     R      b     S      b     T   
   b     U   
   b     &      b     '   
   b     (      b     )   
   b     *      b     +      b     ,      b     -      b     .      b     /      b     0      b     1   
   b     2      b     3      b     4      b     5      b     6      b     7      b     8      b     9      b     :      b     ;      b     <      b     =      b     >      b     ?      b     @      b     A      b     B      b     C      b     D      b     E      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �   
   �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �   
   �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      �I     �      b           b           b           b           b           b           b           b           b     	      b     
      b           b           b           b           b           b           b        
   b           b           b           b           b           b        
   b           b           b           b           b           b           b           b           b            b     !      b     "      b     #      b     $      b     %   GCOL                        NL21::hvac_transmission:NL23                  NL11::battery                 NL33::pv_farm                 NL33::hydrogen_to_power               NL34::lost_load               NL33::curtailment                     NL42::lost_load               NL22::curtailment       	              NL22::hvac_transmission:NL41    
              NL42::pv_farm                 NL31::curtailment                     NL31::lost_load               NL42::wind_onshore                    NL33::hvac_transmission:NL32                  NL11::wind_onshore                    NL21::pv_farm                 NL32::curtailment              
       NL11::ccgt                    NL34::wind_offshore                   NL33::wind_onshore                    NL31::battery                 NL31::pv_rooftop              NL12::lost_load        
       NL34::ccgt                    NL32::hvac_transmission:NL31                  NL22::pv_rooftop              NL41::hvac_transmission:NL33                  NL22::hydrogen_to_power               NL11::curtailment                     NL11::pv_rooftop              NL13::battery                  NL31::hvac_transmission:NL33    !              NL12::hvac_transmission:NL23    "              NL23::pv_farm   #              NL31::pv_farm   $              NL23::curtailment       %              NL34::pv_farm   &       
       NL42::ccgt      '              NL41::hvac_transmission:NL34    (       
       NL12::ccgt      )              NL22::pv_farm   *       
       NL32::ccgt      +              NL21::hydrogen_to_power ,              NL32::hvac_transmission:NL33    -              NL32::hvac_transmission:NL23    .              NL11::wind_offshore     /              NL33::hvac_transmission:NL41    0              NL13::pv_farm   1              NL33::lost_load 2       
       NL33::ccgt      3              NL41::battery   4              NL13::hydrogen_to_power 5              NL42::battery   6              NL33::wind_offshore     7              NL41::lost_load 8              NL13::hvac_transmission:NL11    9              NL12::hvac_transmission:NL11    :              NL32::wind_onshore      ;              NL12::curtailment       <              NL21::lost_load =              NL41::hydrogen_to_power >              NL23::wind_offshore     ?              NL41::hvac_transmission:NL22    @              NL13::curtailment       A              NL42::pv_rooftopB              NL31::hvac_transmission:NL32    C              NL22::lost_load D              NL33::battery   E              NL13::hvac_transmission:NL21    F       
       NL41::ccgt      G              NL12::wind_onshore      H              NL42::curtailment       I              NL12::wind_offshore     J              NL12::pv_farm   K              NL23::battery   L              NL41::hvac_transmission:NL42    M              NL21::hvac_transmission:NL13    N              NL22::wind_onshore      O              NL32::pv_farm   P       
       NL31::ccgt      Q              NL23::lost_load R              NL31::hydrogen_to_power S              NL13::lost_load T              NL22::battery   U       
       NL23::ccgt      V              NL42::hydrogen_to_power W              NL34::battery   X              NL34::curtailment       Y              NL11::hvac_transmission:NL12    Z              NL23::hvac_transmission:NL32    [              NL21::hvac_transmission:NL22    \              NL42::hvac_transmission:NL41    ]              NL41::curtailment       ^              NL31::wind_onshore      _              NL41::pv_rooftop`              NL32::battery   a              NL23::pv_rooftopb              NL23::wind_onshore      c              NL12::battery   d       
       NL21::ccgt      e              NL11::hydrogen_to_power f              NL41::pv_farm   g               h               i               j               k               l               m               n               o               p               q               r               s                  �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �   
   �x     �   
   �x     �      �x     �   
   �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �      �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     \      �x     ]      �x     ^      �x     _      �x     `      �x     a      �x     b   
   �x     c      �x     d      �x     e      �x     f      �x     g      �x     h      �x     i      �x     j      �x     k      �x     l      �x     m      �x     n   
   �x     o      �x     p      �x     q      �x     r      �x     s      �x     t      �x     u      �x     v      �x     w      �x     x      �x     y      �x     z      �x     {      �x     |      �x     }      �x     ~      �x        
   �x     �      �x     �      �x     �      �x     �      �x     �   
   �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �      �x     �   GCOL                                                                                                                                   	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \              NL13::hvac_transmission:NL21    ]              NL32::wind_offshore     ^              NL12::pv_rooftop_              NL32::hydrogen_to_power `              NL34::wind_onshore      a              NL21::pv_rooftopb              NL13::wind_onshore      c       
       NL13::ccgt      d              NL23::hvac_transmission:NL21    e              NL12::hydrogen_to_power f              NL34::pv_rooftopg              NL23::hvac_transmission:NL12    h              NL33::hvac_transmission:NL31    i              NL34::hvac_transmission:NL41    j              NL23::hydrogen_to_power k              NL21::battery   l              NL11::hvac_transmission:NL13    m              NL13::pv_rooftopn              NL41::wind_onshore      o       
       NL22::ccgt      p              NL11::pv_farm   q              NL33::pv_rooftopr              NL22::hvac_transmission:NL21    s              NL32::pv_rooftopt              NL34::hydrogen_to_power u              NL21::wind_onshore      v              NL21::hvac_transmission:NL23    w              NL11::battery   x              NL33::pv_farm   y              NL33::hydrogen_to_power z              NL22::hvac_transmission:NL41    {              NL42::pv_farm   |              NL33::hvac_transmission:NL32    }              NL11::wind_onshore      ~              NL21::pv_farm                 NL34::wind_offshore     �       
       NL11::ccgt      �              NL33::wind_onshore      �              NL31::battery   �              NL31::pv_rooftop�              NL32::hvac_transmission:NL31    �       
       NL34::ccgt      �              NL41::hvac_transmission:NL33    �              NL22::pv_rooftop�              NL22::hydrogen_to_power �              NL11::pv_rooftop�              NL13::battery   �              NL31::hvac_transmission:NL33    �              NL12::hvac_transmission:NL23    �              NL23::pv_farm   �              NL31::pv_farm   �              NL42::wind_onshore      �              NL34::pv_farm   �              NL22::battery   �       
       NL23::ccgt      �       
       NL42::ccgt      �              NL32::hvac_transmission:NL33    �       
       NL12::ccgt      �              NL22::pv_farm   �       
       NL32::ccgt      �              NL21::hydrogen_to_power �              NL32::hvac_transmission:NL23    �              NL11::wind_offshore     �              NL33::hvac_transmission:NL41    �              NL13::pv_farm   �       
       NL33::ccgt      �              NL41::battery   �              NL13::hydrogen_to_power �              NL42::battery   �              NL33::wind_offshore     �              NL13::hvac_transmission:NL11    �              NL12::hvac_transmission:NL11    �              NL32::wind_onshore      �              NL41::hydrogen_to_power �              NL23::wind_offshore     �              NL41::hvac_transmission:NL22    �              NL42::pv_rooftop�              NL31::hvac_transmission:NL32    �              NL33::battery   �              NL42::hvac_transmission:NL41    �       
       NL41::ccgt      �              NL12::wind_onshore      �              NL12::wind_offshore     �              NL41::hvac_transmission:NL42    �              NL12::pv_farm   �              NL23::battery   �              NL21::hvac_transmission:NL13    �              NL22::wind_onshore      �              NL32::pv_farm   �       
       NL31::ccgt      �              NL31::hydrogen_to_power �              NL41::hvac_transmission:NL34    �              NL32::battery   �              NL23::pv_rooftop�              NL42::hydrogen_to_power �              NL34::battery   �              NL11::hvac_transmission:NL12    �              NL23::hvac_transmission:NL32    �              NL21::hvac_transmission:NL22    �              NL12::battery   �              NL31::wind_onshore      �              NL41::pv_rooftop�       
       NL21::ccgt      �              NL23::wind_onshore      �              NL11::hydrogen_to_power �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �                                                                                                                                              	              
                                                                                                                                                                                                                                                                                                                                   !              "              #              $              %              &              '              (              )              *              +              ,              -              .              /              0              1             NL13::hvac_transmission:NL21    2             NL32::wind_offshore     3             NL12::pv_rooftop4             NL32::hydrogen_to_power 5             NL34::wind_onshore      6             NL21::pv_rooftop7             NL13::wind_onshore      8      
       NL13::ccgt      9             NL23::hvac_transmission:NL21    :             NL12::hydrogen_to_power ;             NL34::pv_rooftop<             NL23::hvac_transmission:NL12    =             NL33::hvac_transmission:NL31    >             NL34::hvac_transmission:NL41    ?             NL23::hydrogen_to_power @             NL21::battery   A             NL11::hvac_transmission:NL13    B             NL13::pv_rooftopC             NL41::wind_onshore      D      
       NL22::ccgt      E             NL11::pv_farm   F             NL33::pv_rooftopG             NL22::hvac_transmission:NL21    H                         _�     S      _�     R   
   _�     P      _�     Q      _�     M      _�     N      _�     O      _�     F      _�     G      _�     H      _�     I      _�     J      _�     K      _�     L      _�     9   
   _�     :      _�     ;      _�     <      _�     =      _�     >      _�     ?      _�     @      _�     A      _�     B   
   _�     C      _�     D      _�     E      _�        
   _�         
   _�     !      _�     "   
   _�     #      _�     $   
   _�     %      _�     &      _�     '      _�     (      _�     )      _�     *   
   _�     +      _�     ,      _�     -      _�     .      _�     /      _�     0      _�     1      _�     2      _�     3      _�     4      _�     5      _�     6      _�     7      _�     8      �x     1     �x     2     �x     3     �x     4     �x     5     �x     6     �x     7  
   �x     8     �x     9     �x     :     �x     ;     �x     <     �x     =     �x     >     �x     ?     �x     @     �x     A     �x     B     �x     C  
   �x     D     �x     E     �x     F     �x     G     _�           _�           _�           _�           _�           _�           _�           _�           _�     	      _�     
      _�           _�           _�        
   _�           _�           _�           _�           _�        
   _�           _�           _�           _�           _�           _�           _�           _�           _�           _�           _�           _�        GCOL                        NL32::pv_rooftop              NL34::hydrogen_to_power               NL21::wind_onshore                    NL21::hvac_transmission:NL23                  NL11::battery                 NL33::pv_farm                 NL33::hydrogen_to_power               NL22::hvac_transmission:NL41    	              NL42::pv_farm   
              NL33::hvac_transmission:NL32                  NL11::wind_onshore                    NL21::pv_farm                 NL34::wind_offshore            
       NL11::ccgt                    NL33::wind_onshore                    NL31::battery                 NL31::pv_rooftop              NL32::hvac_transmission:NL31           
       NL34::ccgt                    NL41::hvac_transmission:NL33                  NL22::pv_rooftop              NL22::hydrogen_to_power               NL11::pv_rooftop              NL13::battery                 NL31::hvac_transmission:NL33                  NL12::hvac_transmission:NL23                  NL23::pv_farm                 NL31::pv_farm                 NL42::wind_onshore                    NL34::pv_farm                 NL22::battery           
       NL23::ccgt      !       
       NL42::ccgt      "              NL32::hvac_transmission:NL33    #       
       NL12::ccgt      $              NL22::pv_farm   %       
       NL32::ccgt      &              NL21::hydrogen_to_power '              NL32::hvac_transmission:NL23    (              NL11::wind_offshore     )              NL33::hvac_transmission:NL41    *              NL13::pv_farm   +       
       NL33::ccgt      ,              NL41::battery   -              NL13::hydrogen_to_power .              NL42::battery   /              NL33::wind_offshore     0              NL13::hvac_transmission:NL11    1              NL12::hvac_transmission:NL11    2              NL32::wind_onshore      3              NL41::hydrogen_to_power 4              NL23::wind_offshore     5              NL41::hvac_transmission:NL22    6              NL42::pv_rooftop7              NL31::hvac_transmission:NL32    8              NL33::battery   9              NL42::hvac_transmission:NL41    :       
       NL41::ccgt      ;              NL12::wind_onshore      <              NL12::wind_offshore     =              NL41::hvac_transmission:NL42    >              NL12::pv_farm   ?              NL23::battery   @              NL21::hvac_transmission:NL13    A              NL22::wind_onshore      B              NL32::pv_farm   C       
       NL31::ccgt      D              NL31::hydrogen_to_power E              NL41::hvac_transmission:NL34    F              NL32::battery   G              NL23::pv_rooftopH              NL42::hydrogen_to_power I              NL34::battery   J              NL11::hvac_transmission:NL12    K              NL23::hvac_transmission:NL32    L              NL21::hvac_transmission:NL22    M              NL12::battery   N              NL31::wind_onshore      O              NL41::pv_rooftopP       
       NL21::ccgt      Q              NL23::wind_onshore      R              NL11::hydrogen_to_power S              NL41::pv_farm   T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �                          ��     �      ��     �   
   ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   
   ��     {      ��     |      ��     }      ��     ~      ��           ��     �      ��     �      ��     �   
   ��     �      ��     �      ��     �      ��     �      ��     �   
   ��     b   
   ��     c   
   ��     d      ��     e   
   ��     f      ��     g      ��     h      ��     i      ��     j      ��     k   
   ��     l      ��     m      ��     n      ��     o      ��     p      ��     q      ��     r      ��     s      ��     t      ��     u      ��     v      ��     w      ��     x      ��     y      ��     z      ��     /      ��     0      ��     1      ��     2      ��     3      ��     4   
   ��     5      ��     6      ��     7      ��     8      ��     9      ��     :      ��     ;      ��     <   
   ��     =      ��     >      ��     ?      ��     @      ��     A      ��     B      ��     C      ��     D      ��     E      ��     F      ��     G      ��     H      ��     I      ��     J      ��     K      ��     L      ��     M      ��     N      ��     O      ��     P      ��     Q   
   ��     R      ��     S      ��     T      ��     U      ��     V      ��     W   
   ��     X      ��     Y      ��     Z      ��     [      ��     \      ��     ]      ��     ^      ��     _      ��     `      ��     a   GCOL                                                                                                                                   	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /              NL32::wind_offshore     0              NL12::pv_rooftop1              NL32::hydrogen_to_power 2              NL34::wind_onshore      3              NL21::pv_rooftop4              NL13::wind_onshore      5       
       NL13::ccgt      6              NL12::hydrogen_to_power 7              NL34::pv_rooftop8              NL23::hydrogen_to_power 9              NL21::battery   :              NL21::curtailment       ;              NL13::pv_rooftop<              NL41::wind_onshore      =       
       NL22::ccgt      >              NL11::lost_load ?              NL11::pv_farm   @              NL32::lost_load A              NL33::pv_rooftopB              NL32::pv_rooftopC              NL34::hydrogen_to_power D              NL21::wind_onshore      E              NL11::battery   F              NL33::pv_farm   G              NL33::hydrogen_to_power H              NL34::lost_load I              NL33::curtailment       J              NL42::lost_load K              NL22::curtailment       L              NL42::pv_farm   M              NL31::curtailment       N              NL31::lost_load O              NL11::wind_onshore      P              NL21::pv_farm   Q              NL32::curtailment       R       
       NL11::ccgt      S              NL34::wind_offshore     T              NL33::wind_onshore      U              NL31::battery   V              NL31::pv_rooftopW              NL12::lost_load X       
       NL34::ccgt      Y              NL22::pv_rooftopZ              NL22::hydrogen_to_power [              NL11::curtailment       \              NL11::pv_rooftop]              NL13::battery   ^              NL23::pv_farm   _              NL31::pv_farm   `              NL42::wind_onshore      a              NL34::pv_farm   b       
       NL23::ccgt      c       
       NL42::ccgt      d       
       NL12::ccgt      e              NL22::pv_farm   f       
       NL32::ccgt      g              NL21::hydrogen_to_power h              NL23::curtailment       i              NL11::wind_offshore     j              NL13::pv_farm   k              NL33::lost_load l       
       NL33::ccgt      m              NL41::battery   n              NL13::hydrogen_to_power o              NL42::battery   p              NL33::wind_offshore     q              NL41::lost_load r              NL32::wind_onshore      s              NL12::curtailment       t              NL21::lost_load u              NL41::hydrogen_to_power v              NL23::wind_offshore     w              NL13::curtailment       x              NL42::pv_rooftopy              NL22::lost_load z              NL33::battery   {       
       NL41::ccgt      |              NL12::wind_onshore      }              NL42::curtailment       ~              NL12::wind_offshore                   NL12::pv_farm   �              NL23::battery   �              NL22::wind_onshore      �              NL32::pv_farm   �       
       NL31::ccgt      �              NL23::lost_load �              NL31::hydrogen_to_power �              NL13::lost_load �              NL22::battery   �              NL32::battery   �              NL23::pv_rooftop�              NL42::hydrogen_to_power �              NL34::battery   �              NL34::curtailment       �              NL41::curtailment       �              NL12::battery   �              NL31::wind_onshore      �              NL41::pv_rooftop�       
       NL21::ccgt      �              NL23::wind_onshore      �              NL11::hydrogen_to_power �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL41::battery   �              NL12::hydrogen_to_power �              NL13::hydrogen_to_power �              NL42::battery   �              NL23::hydrogen_to_power �              NL21::battery   �              NL31::battery   �              NL23::battery   �              NL22::hydrogen_to_power �              NL41::hydrogen_to_power �              NL13::battery   �              NL31::hydrogen_to_power �              NL11::battery   �              NL33::hydrogen_to_power �              NL32::hydrogen_to_power �              NL32::battery   �              NL42::hydrogen_to_power �              NL34::battery   �              NL34::hydrogen_to_power �              NL21::hydrogen_to_power �              NL12::battery   �              NL33::battery   �              NL11::hydrogen_to_power �              NL22::battery   �              ag     �              8f     �              8f     �              9v     �              �d     �              �d     �              9v     �              �     �              �     �              'j     �              �t     �              �t     �              �t     �              9v     �              �     �              �     �              s     �              �     �              s     �              9v     �              �     �              �     �              r     �              B�     �              B�     �              9v     �              ��     �              ��     �              9v     �              �h     �              �h     �              9v     �              �     �              �     �              �p     �              �     �              �     �              s     �              �     �              s     �              9v     �              9v     �              9v     �              8f     �               �               �              electricity     �              Y�     �              Y�     �              ��     �              Y�     �              Y�     �              �     �              Y�     �              �     �              ��     �              Y�     �              Y�                   �                                                            in                   out                                               	             lon     
             lat                                                                                                                                                                                                                                                                                                                          !              "              #              $              %              &              '              (              )              *              +              ,              -              .              /              0              1              2              3              4              5              6              7              8              9              :              ;              <              =              >              ?              @              A              B              C              D              E              F              G                 ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   x^eR}P�~���:c7�;ɽq]�''`
��!B�� �� *:0�o/C@AZ|��c��LzNz�8"厉oX�i	��3��o�����?�{~_���~?��"s�#ӈ}Q����8�{��֋��fv�B�%�3���7W�"U��a�����; ����naG�C���\���Q�=�e��>�"[9�_p�GS�������f�O���{��� k��O'�i�C�c����JT ��
�|�O��@����îA��
�]��t	hP�#��Q���|� {M��*�&�F=�@Y9��i���t6��^�A���ϳ'�š���pE��Þ[��Y������K�_�^Gjt[�Uk؃hg�«8W���ٸ��9lXwnz��GPK���?X��'�-J�`;��r>�?���
��c���@=���U�b�5q�Do�F��w^�/ɢ(�3?����Sa�ex� X�I�#�5�,*�9?T���r2I~���=YpU8�R���1$��ĹL�:p��O�\@6�}�:;����^|	X��
�^(�����|��T~S>e+W����	��օ�E�G��{�E�<��o�u�>�Df�y�[�8�A>ȟrwu�|K1�v|�M�-?6�`�g�}�lw��C�x�L ~���.x�����B�6�4�u��*b�� ��6�6�<ü��-��Ȉw�I��ﴣ�0�dW3�y�Z��=�Ja���`�v0m`0a�H����1���N���ךA��<^e\��n�LD8F�Lr�7+�CN�Lz)F:cړ!Sv�f�oF�G�U��a�D�AW��ɈTY���Q2��(2s�J��-��LW���b-�JI���� =^SM�GߕV&@8C$:)��B5�DS�I�zӤC'���	�T��w��
�g��S����e�o���%�?^������������ri�^��ax^s{�+��Vܸ4�D]���on���9���T9��ar9c�^�W�������y~d]�7�sr7':l8����S�o�[�c"ۺ��[��|�-�=}9k�Z�u�������M��<����\���o�Œ���4޹1A��QYW�T��+9����T|qf�S����d���5��L�1{U�ÛUwp  �)`x^��c ����g3~���7+/� ����o�$l�(tEn^���猘��'����$���(�

��b`0x�nZ��X��-����M�̛B&�P�������5�;S��b`P��\~$1!�5�c^�Y����ês\��o� �[��9w����������7/����=2��"�;S�x^�d �s��4�c0��!��ׁ��i��LA�Ě+O�N��ު�tה����Z�n1��\ƺ��4�-Q��g����N��`:H��9^�#L�E����a`���櫿 ��~����L{����J(�����I��p���?�g�([7��A��B�ӥ@�R]VU��U\�XU���Ǔ��v����=p��� �kTwx^��1��t�E{ ��   ��          ��          ��     
     ��     	     ��     �   x^c`�   �    OCHK   �     �      ;    	   calendar                     proleptic_gregorianq�                      	                                                        !       $       '       *       -       0       3       6       9       <       ?       B       E       H       K       N       Q       T       W       Z       ]       `       c       f       i       l       o       r       u       x       {       ~       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                                          #      &      )      ,      /      2      5      8      ;      >      A      D      G      J      M      P      S      V      Y      \      _      b      e      h      k      n      q      t      w      z      }      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                        
                                                "      %      (      +      .      1      4      7      :      =      @      C      F      I      L      O      R      U      X      [      ^      a      d      g      j      m      p      s      v      y      |            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      x^��1    �Om	O�                       >]     ��     �     OHDR�         �       �           ?      @ 4 4�      ��     X                   �8             shuffle            deflate            ��     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               P       DIMENSION_LIST                                    ��     �   ���OCHK            �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   ���       �             ����TREE  �����������������                      ?�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OHDR�(         v       �       v       �           ?      @ 4 4�              �            �8             shuffle            deflate            �     v   �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   4       _Netcdf4Dimid                          �f�'OCHK            �4       _Netcdf4Dimid                          9��UFSSE �k       �   �   �     �   
  �     �	   O �   `}
'P[��OCHK    Q�     X       `       DIMENSION_LIST                                    ��     �      ��     �   +��OCHK            �4       _Netcdf4Dimid                          �H�TREE  �����������������                             V             v       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDRM(         L       �       L       �           ?      @ 4 4�      �     X                   �8             shuffle            deflate            �&     L   �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   �l�QOCHK    %�     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      V�             �B�uOCHK    �q     �       P        _Netcdf4Coordinates                                        
   P       _FillValue       ?      @ 4 4�                               �F��<P%            ��o�         Y35OHDRM(                �              �           ?      @ 4 4�      ��     X                   �8             shuffle            deflate            �        �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   ����OCHK    r     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      c�            ��            P%             Y�\?OCHK    *     �       P        _Netcdf4Coordinates                                           d�&                  x^�	<U���?Qd/kE�TTDJmT�$����$ZTB�*K��B�BѢM�%���T$�R*Q~gfν����~�����Y�k�5�Y�rΙ�5���3n�c��8�}�]K2�'7]��f0�QA8�q�C���&�y�  ��,����J ����i��N�m	)�B�f�b�^�� �K��W��Je 84�7c]���pT�KR2L�C	%��	XǦ� �-�ɠnZA)�Z
�	�D0�;H#�Q8�'3��{3 �Tе@dJ<Q$U��&x��s4�{��p�<c��_g�&3��C�(�����������H���tQ��{v���M�g �Q+8��6ed���350�� )�l��!�=Y�.V����zfC���h|o�0S���/�:̴�#L��;��&-����Zl��ޔ�F�iy��H�� ]Cr�DO�{�ޛ["���\+��ZJO	��*�VÊ��ȟF/��}�ؠ���¬��܏"[��z�!zbt1�����3�l�� ΉL`"È�L3!x�L��3������iBE��lm`6���ei��>Q�A@�*�'�Q�D�pWd����41�Gv��l��e�4�0޻�q�K�cu��^j;��4����Ϩ���Ǽ�?�q�܆,Qk��%z�T�RaW������'C3G�<Vj:x�k�l�Iy�Ց�&)�yz��Um���J\_��L��s���۵o_9Vک{Lxi㘅���Z����a��.���\��Q��ekG���j3,ݔSyljU�Y^)��B����y��{"��TI�ge���׮�?2*V[�{���%��.,�q�gJ �}@LK�`����ޮi���!B��) �z�1���(g�N0��w?.$��ud��{�F!!� ��D�a�K�����\�P�$�%��?^_&e��h�����w]�/3AI�\$�YP:�\6�،�JEċ�aK"� #X�{;$� �OH���?��]�D�M%~ם���II��Hh�M+��$��^2)'�&)�35� \��Q��@ ���P�F��EM"x�Htƴ��dj��d���79�O���)P��0tΝ$?�9 v�k�a���m�n�-��<D����m����iֻ����{�'^\bViU�ߡ�v�Ʀ���ow�������̹jb�
�M�q�w�]��B���U�wV���J�T�
X�FP�qos��._�9>tm?����x��G"�R�W�*,��}=�}��G��)_��o�~0�u������\:�;��E�QG=N�;5���ť6����n��q츼?�l�l'�~B��ov�J��O���1o{�=׸�K�D_��eެ�]����\�����q9���@�^���dy�mʈë8Ne�0L��dzn̧��2�(l�F��ښ�π�f��q�<*� ��X0
%6`�(a��9����diakXi���؆y�3(�C��S
(��ǉ���ڽ� �H-DZ�ɲ�wo�u1��ǥ	r��`c�y�@V\x�*��ד�,Z��n���x&���YH��5H�|+��h�������^�$���"	d�e�+,�YLy��uD��9��'6�������<�=�۾G"�Y���`'J�w}�+$�q;-�W�v�>B�E��t��1(7�3Fq�H��sr�î�+8Dѕ�]'���4{U	�a���1��34B�_݈�&

���<3V��h���jmN?�(�Q#�\����n#M�<�fd�d?O�i�2���

DbO�؜���#^�5���E�{qMS<���!Kf�� #(N�dQո>���b�p�¹#���s0Ap�_�:�=��a�>$d C�� H�?j* r���x��f�N��7T�ަk0&r�ކ���@ǖa�촐XQ:<��t��(aV����ʗl�"�^6{��^v,f�ʂ�*�a��x��t�)�BlA�`dec*� &8���3�P�K1c(�ɜ���E)%���`=��_��qXΘW���Ɩ��"��H3wƋA�F�X�"{7E��XU�ԑ�{O��W p����Q� b��F����Ϝ��>#9m�\4�y�Nb�A^�_!v�����Mug���=/�aJbS��h%��j��
��:�C�׈W��'���@�08$.�%'~0�m�𞃗�]��Gn�*qw���K�ykX�N\	��ßv�+нn��5�׬��ن�/_�n�P\�r�d.�U���"��J� ft���,QM��:�ӕ� 䪢� ~p��8�Yx�!a����5o�&�E�6��*���ذ0n��B�lx�� /���)(�9$1��| �sbo? �p�_全�&��t>�Qd��S�!�CB
�����?ú�:�g(���Φ8#E����P�^���wJ��!9k�����	���xP3�d���Mq�^x�������FR5X
��1'�W��~ed��bf
>F����]��hY��koFK���Q�~-�h�3������d��7��|����u@r��Dt��݄��b�h�� x#dz/�|���k'�)��.AD�Q"�X�V��R(�_��ZQz?�4�K�Lˣ��Y��>C=
Q@&��i�j���w3θޅ0}�Qȷp�E����C̬�T�Q-\��nV܌/:J�+��5���r�[-�|��m�2BY�p�X�5�@����i�+�#-k��N�z��l���}�Ϧ�(��,��D�tm�>����Lב��sg:���}T���;�-;j�.�k�'4qɳ�:e�"����)���J
�&��� 2�z������1
��i
m��������bŀ*�����h�j�EU����}�	6P����c֏k�Y��j�[��j��#�`{���b��9K��G��_x���0ʄt|[�D<�#g%P>��W�9�D�$VM셛3l�t}E]��8i�\b��2�C���:���	���E�5�>��*y�an����:�a�{�����s������W��X����!�6\;O�JO6uo�94���44��eZğ '9
���
��~[��N�	j�mՂ��]���mB�5���z����\/j��:Hѳ�<�XV���m��H,_��Hi8~\|&�"�,��m�����Ǯ*��8�N�r����I|�*-\�ޜ��1񥣲���ܹ;�̽߇d��
Y!�<��q����n���⢰o�{�L�����=۴���h�煪��k��Q��c<sb��S������[{43���M�w/���\�P�k�����2r�p�<�A~ꁨ�ZNO����xe$Nܕ;��^y��!���E
������uꃨ���u�r�����}:f�됧��3�W|u�Ӷ�z輝���K��!҉��ʯ��]�@|Γq�r^�2����%X�!U^�|l����Ñ��3��{L���s�o��d��$A^U~u��]Z�g�<�޽�Wy)�~ڈ�P�jj�U���H������qd7��<Jw㞅���v!��~��u;P<fT6j��eǑ^�z�0 \�b�J���;1�53D�������?vl�: 5}<O�N�$~��k���sM�k}ԕq+}�cA����`Q��5G���l��w
�ۮD(Ĝ_��vҺ8�x�$�SD���'T�h:f���*��ͯ�)ST$��֖YL3%�	����DA�������)�7�&U��_1�t�O����q��R歟�����7�1�9T粙�E��>�9��^��l���vP~M�	�#���)�>�YS�/�q�t��\.o�힅[��p`�ג��^�綏�ͻpԗb�ӯN���g^��,h0r���ݜw����,���!�f�g�FK�eX:Zuϸz��ʺ�Ԉ�D�ڟ��'2�������wHJ4x��ǵ,��F�������ۢ�̐�N�|���Η/BST��T�S:n{C�x����c��/u~J-�6}VظA�ޱ3�G�������]��y.����\���r������b�y��*�ꤳk܁E3M�t��5��d�K�[���JO�����/*7P�Sq}�{�,�k,�6��j�yiZ���5Y���ښp�c�ܥ�p����~{�Z���v�+��.�{kT�K����^t�I���:��!�r���bn�H��.�+2l���]�7~��\:X�ι���ae��[|�t`�nZ�e��~��>-)0-`S^]����u�N���|/�3�G��QW5Ј�3n��1� X�O���q���q#B����� 3�-��.����B�ѵ�@�@��x�(�=�g3��I����8�an���7�q���Vq��'�ve���F�	�M��FQ���ܯaj�'逸��	
����5�4�Q�����?��|J��9�Ҙ��zB
1������%H�
��������,�~i�m�N��c�S��XYM�so��:o��K�n�hTV��$5+y޼h�A��s��kw�Xb7�pE�oP+������X�Z �Io~�����:E�[�.5�Y��(�7[�&p�=���'�;F�7�볏,N|\/�W�c]�r<�ꊓܥ��鯽�"�����+�)ff?O
����6k�@+�\m�ԜS�%�6���8>暵��p���鳵��emi��h2u�����Ɔ�o~.�z�w��o��fuW�#�O�@'&7C�\��mN��h_��f��.��2Z�z�܄�N��M;j&��l�Ʌ01�d�K�E��׮/��@,��-����e�~�� /������1�Cu�g��y��V$���U (�u���>���T����\�g���B�E�+\��<ˇ��NIw�Pw��V\>١�1Ê^�s>]���1�A�4˝F�Yę6�v^\3j�i�����\�&�N��o��']��I,z��x�ޫ���v혛�B���Ύ������>�hN�SOX6]_u�����IM����T��dR�/�hu���hל�dx�����n]+ߦ����V�1���������32��8']�C0O0"̽���Ā�׵Wk��1q킶��d���w���2
��0ѭ�Y�����V�B�#F���[ojjZ������6�DjȚ�N�;�)�N��I�ŷ5%��n�ݴ�B&l��7���djI�1	1gy���c��:w�à�n{�	2�v�[>����wH`�$!#�
��3�i���L��U�`���/�m�dc&�q�d���ї3L���^J2'
��?F��H����[s��,1]���-�s-�V,�+g�8����ݸs:bf!�>$�ǝѩ��#����d��dϋ�v�����ڂ�V����L�}�y���+Y���W(�"�Q��-l���+P0��Z	�P�KQ���@پ�{2����ظJ1�pToo�?6�W+�\k�Cv��7Q;s���~�(..V���!kCy�67v���)f�s�#�B	4&$�2`i�{��c��'��4���
TN!iu�icf�Ủ&?��o�7���'1�4��sARFr0*�@���&�3+ċ�2����g�jd��w=J?��5ʮsn�U����N(�q����c�o���)��}Cz+��P
I?B��O6t��'�q#�,���ʑ7�L����I���`�MK�~��+��D��/�tya	W]�A�R .Ĳ���@kt8ĺ.�)�i(����'R��z���KwR. ��H�
F͌Bf3�24,�?�lß3f��Aq]�����F}ea�����a]�̤ƆL�yv���_�̙3���C�Y���'q��5���3,��9NX�G?�:޿�;���Ta��l��/�dE�e$0�4=�=
0N�>qlz�V-��u�2Ü�)�J6�ۇ�[1󪜰��W�W��{�P�KyyQ�sT��O�'7�Yp\3��T���7Y�w��ƪ���9W3��n1?�����K'ϻ>!P�V�YZ���&���F�g��;��:w`�OC�oc3�/+������\�t�(����������v�J��Qqj�뜟��{������bIޞ����%�4?�ܠk !�{h��F�}�U�q�n}U:�hm�������}�:���A�𗽋_�*�\�[�Y�N���߁��/���d��:{�c/�A�;pr�(�Q��⦉(�(w�=�A�5��*]�n�M�v����Bv'�� ���K��.Uu=\�_?��j�"�&��N����~��-~���'P�;��}�n�a��/y�P� ��O����oˋ����ȍÿ�/����Ry�H:������x}��8����T�?7��3��@� �X~<�J���7��ࡨ�sO�?���4�[7���یic�a��/	HB����Q\��=gFc�!�?��v���y��q��2��0�A�v��H����r��v$��eg }��qP��B�V���J^�؃߼Ow�_�Fr�^�臽�ǿ��d��H$��U1C���F� ���kR��o��6�O�vEi1_$Un��y ��ը��_���н5h�l�!�����WVˠ��+�!��^�E����ץ~�S^e�/��y} ���C)4��b�Q�W7x��$��y�VY ��}4�ԟ�h�y���;<<���'A��t��G�'*��q~k�nV��#���+̯�me��ya���\GS�#��=l�ĩS Ih�<��"�A���An��ğUK����{ځ5pBZ`P���f���ۇ���zQ49�-��%���c��J�����Y1Gn�t�@�d�(�;��bqג����3�E��p{+F�F���L�crDӇ�,=�<S�ߵ:����M?c�����RT��|N��zה�a���k7.(�I]T�#~�����B�X�m��C<7�3�9^��=xB��-^[�9SH�*"ѽ"Dn������
f
lKt�N<C|�c��W�*u՗ۆ��6i<^z���]�R��b�����^��R(������Ο�~�3�y���RC^x�����������k7ݙg�Y�x����O�S���;����"�0+zF��3�O'ՖWF'�'�6[9�F�Qسkn�o@s3���jK�̀b�(����� �����X�4�K��w���l2���OF�����'��V�*�Q
]�k�8+SGk=L%E���[Cgh��e��V�V�үR4,X�p�;Z���7���n�o,�vPk�;-:&�6�&��*.�鰸�}��_��ܛ?c
 �W��/���#���l8 "yP��cen��(!s��Xh(�|�5��jZ�-��mǁ|B�=&m��e�ׇ�!��F�r��6��@�[����Y0����Ұ7X�t�� 7��g1�>���G��F[M�*�%�,]�ZX�^�� ���{V��yo�RQU � �l�-a���;l��:�6^/,(,�)�L���_�s gc��s�����W���ֱ٢l��(�v �dү_!��_�a8Z��'
]>�Kב�+O��=r��@j�cx6w�Yl幌G����������8��.���,�S�Q#����O��)Y � 0O�m��M�K�[��x�,➴��3����E3�픽�'���x���ZZ�޻�y���OH�bΙ�E����'�^�cq�j�ʌi�jTjw�h���<r������㖡NmFj⌨�!�����=�����!>ň�n��[\�Vn�)ۤpm��E.��e���j3rV�|��4@ƂW>����y�$֍uϯn,���W��a}�D�;.`��<�yuK�ڍicVfΚ=7A�����A��ޥ������?
lz��R��&̠��V���m�/���o�r��证� i����CLc�a��.��%3XB�Q�ğ#��'���H2/�#�a�T0��|d�,QNDF����#_;;;�����<s��Y9�t���2����&�LF��j��u�Y*W#{m�JHU"l�6殼�i��׷�B!�Z���~�׽�}:�d�K��^�_o�4���G�|�\P����H��#d%~�74<X�C\����Q��.U�V�
fޱC�㚓��
U�TAY�U��������B�KT�[�zu�U�o�<��pjW��K(�J�o�LN��;���t�?�>���A��۾MÉ;�b�>��P�P���(ꝋ��k�p*�2p�m����\�+�/�g9����0#dKt�+��:G5G%`׻[�"�9�|l��Ӓ{9�����Gk(�;�){�)A��|��ں*Hc��Ś��}&sa��0?EO�L�����*|@� f(fS �oˑ�Sl�ŗGyj�U��^Pr�j���+r.�Int�7�6�JpϚ�[�L���Rǩ����s��(jV%�2ZռԻ�T�c2R/V��L��hٕ$�a���qK����Nְ#��lpT���Ѐ�-��Ez�D�V�z����}�a���橞����O���KYk.�k]sɪmz�gwg������i��UС��9��������Qq�tUǕש�^O��Բ6+^$��������}�{]��!p��K-���_^������eg�������H2J�����#���d��
'��,|��W_[0$�Q�2�G�*�[��fm��S0cW1�3w����9<؁����&�e
z�q�h����-���TZ���EJ~�7*k4 '˯�i�d0COP�$��t��h�U9��l��2f!�2�2v�MD��C=��;�q��]��v����E���gd]�$[J�C���1�w���f��І` 
�Ӵ�c��ѭ��da^_�����Ðz�����z=��=����� 5tw�i��XM�|N���"$�D�gsd����'��.�<��{nwy)�r�,<���A��X�'��DdW}c�t(3dTQw��\N��S܎��X(�{P�������p����3��tm ����Q�ϗ}���~��m����`�}UR�q������,���7_hͣ:�z��lGҀ×�b��2���]vk��3�N/��0��>���v�[�P9O�V�ʂ��w��b�>5}�q�5�����V��.�z��ZO	��5����ܨ��Ӯd>�}�L��ky}P�(7lylW�`�R���A�9�.��$[���{G�R��#j;^���Yl6���( �|ɠ}��7�?<�������_^���4[������廳mSc �G����a͒$ky�� ������m��������s%%��ǻ�>:�ĳ�w����;�ߐ�6�2O�+���^��3����@�H�#��_�B�=9؅5!��z|wi��1�D��������b�/��c�n�7R�n�Q�PZ�4�u�~NZ�����B����Vt��s�:��x��_B�`b{d=��B%�I�:e˯}@������`�������q m����{}�6
�+�H.� vJmg�mȻ��7U��|ڊf(��3Y����+{�-�m�,ВÒ���$��)"��,��S��#?��o�,󚏤�$޷��r�ڥQ����ƣ�7�4TNޮ�����H�܎��̍�z�Q}xl���:g�uv���v�,{A5ZIN�P&��"L�q�*����k���X�,����5]�+X3�x�ƚ�`�:x����R�J�#�P]����&�������"}2�Nb���H#8kI|E�(
���+��� �aO@Хj0 ��	{�!�c޴]Θ���}�=���m|+L%����8�=���������G��q� �B�=C�+�w��OK��GW�Jr��:���^��q�[>@y��`eH9N�v�>=ʵp�������?���Uńn@c)�6��by�m9nS`BA'8���)�x������X0*BȌ';�¢.>�8|і="����'�v�mk�S?D_I��0_��7XN~�	?Y�\��|z�'�~����}G��C�������m�e�n��д?�%A��,pz/ߎ��Ħ���"Xq�� %�-�3�jOB��U,�>���&*3�R#���|v��Y̚ uI��o����?��M%���Ѷ��	��uKBxq�> �k�����+N|�U�\P%����aT$b��6�(���zn���<��=�����l�D;��-�����X9�{����&2�߻�4�pdB��Z�K��jT֢�Xq�60�����4���|C��h�U��MJ�ؑ-��bW>�_uz��~c�Z-���Vo�;��]̗����o�}:����=���U-�h�"�����B�'WD
�[g.�-�{wUP�l_m����j��,��q���r��1 ���>���j�z�n9,Q�o�<q�N�S��كL��iF�)��/�rm�G�{�fޛn��F.���%l�tq��$��E��h��d�gd���x�e;�O�`��&�1�z�&{�=j}Vt���������5��� ��L�ɻ�i���e��d�B�f���<�K)�!�5�������D��O� V,F����1mt��ʌ��P������R��8�1yFK�|-�������fޝ�K�%��9����M���q`��=�D|�8�'nI����I��A��^�F�.L搿�F�����b��#�O�`���Lǆ>h|�Ŋ1.�Lj�	 4��� �-Q��G7�2� q���k�sU���uXI_'���g'	xB!��	����HS�,�D��iW\�����#e��kt6�~3)�^M"�b�0���]������M������ױK���O���R��t[����e��]�\P��x_@:0j��%]U��F�|_1��R�pmĂ�Q�j&#̥���*�|��Q��v�y�)9>����/z���hZֹ˨ߨ����l��
kw���z#���-�!����`��a��L��^��U�%�*E������3��r��ܛ�"�g��T�-Z�ma�V��w)SV�� g��4����z�ҒI�u��)#��8�a�0�^��9@�,�2�(l�FՒ���π�f��q�<�� �ԧX0
e���u�0���`Z�����a�a�1MH;t�\PWh̬��}���'v0�T=z�'C�Q0W��������XD��<�at����R��j������}�"��A��`��m�9~�@M׀�x>5�;$N]�=�Bڱ���t$A�׶�� ��p��H��;�Tq�!v�м���/P�$���x$�����I(}��j��7�	>[W�v��C�\Ѐ��.�C�Nj=�(�= I�^^}̩��І���^��K0��r���0y�-�����]����_��������'n��[AURh�/�Lcҁ{$�@s8�gO>��>��g!:!q�d?O�ir L!���k �3b3�P3'���Rү��@��Qv9�dF����odŉP��G6�,�NZ~@�\2���A��u�#{�7D�����$��L���==������`����
~C��s�3\��ۈ��k"���a�촐DR:<��t"�"-�������d��	��������c��+:P�`����'�)��
������������8?C	����%3�sٿ��1���k���\��c�m2���[
d6�[���~�����k�|dlq�w�����݌b vwR�ĠԒ�Au�u|M����]	x�J����&F9G����r`7���sw���!"�E$F��hC�1J��Ծ��+�ZIU28��v �+Fv�h��2���À�V��V��T�E�4��H����j#��o,�s�YT�Š7e`4�0�m�-�tZ�D]q6���n� Aܵe�j_?%�t�r ��e�H��O�'���NK`C]��p�-��s5e���&m+�#�k�ր�h#���Sv������`��W�\0e"�
��kޣN��"x����`t���%�#�d�{��^�����rBvp@�_`��:[��2}���<���E�i^'�HG�����|e���~�4.���Hn9쌾Rc���c'��A�M�n��b!���L,�w��s^���3h����2d'����oG�]��ƏU��_�XH�
J��蒽��տ9� m�(7�����W��Ӑ�Ö�H�z�$�!9@����[��dW³ OXp�L���`Lq��0��F!خ�dQ"�pNd�!�`+��uᬮ���L#n��4�o
Գ2�FFvA=
Q@&��i�j������,��b���ԣ���!�~)G���)Y ��gc�=�U?��o�_�����E�/H�x�d���.\p��*qw�7��c��<��M�u�d.�Ww|�҈�.�ڪ�����x����*^Wu�<�����}��"ͣ�]U�:�n���]ZG>:�R�~6�E�O�}��0A������T��x?���=A�YܵT�S�w�e����[����دyX��Y��ι;���p�_�z�P�m�Z�AI&��O�ވ��s��N"��|gg�Gu:^�k�='|K��q����ţO�/��dwe����ߡDj�à��3����\�Kb��^8�^�W����Е������* BPUQ򨾿P'Ô�<���
+$��w�v��y�]ޠJ���Ȁ@m�����ٗ4�h��7����閁�.C�*��5��0�|t���������	�����s�^��a�bT4Zu���6Y3p"�npd�����P���u�J��!�8�G��u/��K��xg���/8-��ã����i�oo�Ҫ�h}>Ssyd��	Yf&n=���QY�����g5�Y��4��W摝|�\G�}W9�)��<da�4練gH=>޼Zg��k�U��_�l��,�'JB|jC��冴�Ct��\�/�z�S�h~�D���[ď~��<{�ٷ��x/���o�fʔ��[]l��i�~�?#��@�,�sW�8��5h�h`�څ��jw������WeMi�ۆ'ˌ5�߂E�rG�?�Y�_��o��a�v��ꔎ6�^x��� �We�w#���f�}.u�@�<����Q�05$Å������>x���E�(���sD�ʑ\� I �W���W �I��B�m�4���W�n#��������*T�ByjH@�
��6ٙ�����bK���|$��>�eb��}�����'�B��ޓ� �B���Ɋ�T͔������_z'���a�^����x�C444���P���z�*���B�d煞��#��a�fM��޾n*�p~ϋ�kn�-+vs����p�J7�YB�B?cb��kF��ST.ld����G�O�R����j���2���Н��';�E.�Y����f��GU�9�IRC���<.EE?۸w��҇�ϔ�k�2�P{��{�%r
�ۿ��z�vNZH�NDqV]N���b2L�$MT:��n��ء�많~px���?����X0G%������W]���S/��]��t��U~V{g�p���&�΄���v�0|����e/b�j&]��oJ�r<��Os�u��ڳ���	����r��>L# 1^\;�p�J��#���*����z������Z]�&���-�g�=���^��7${f��tȿ�tM8e�̑�i��Vn�t���6Lt�O�܀%�77Jw�L�>�6W�0�-�.�r����ڂ���m^JL�V=�ct�RI�����~���o��J/_,�#2��Y�ǯl�_�:���Ǜ$~�?*��p�財I�´�����ty�}����O�ڍ�M��7�ʱ͚ScT�����*��%"�r��S����髛/�r�eu?~��牎.��8Y��N��uwL�O஬���s��+A���D���q���w�o��_��{A�Y_�I�\A�:s�'���o���J&_tC~��,<qP8�Zf�ʞcc�@�a�)8����'��c����l.	ڹ�1?��l�o�[��p/�����,�\�j����o����]�&
R�_>9�*�����t��?C���������M��|259$/o�JĨD�XH���u�^��o�� ��2��%�霏��/�ȗ�b����"��y��{���K9�+�w^<x�a�\���@�o�����e7B�2�8(��/�|�1T;�D>!��n��S��A�B��h�����jؕ�\�ު��W�r"5a���=/�獚`#���h�d�,5�ͷL����ǽ����X��3�MNz�k���ԃ�>�,�X���'a�d�7���&K�mN.4:[@�~D}����;�)-q�_JԬh��s�O��>�:��S5���<`xƻ�m�x���X�.�������Ԡ��l��2���(�jPjl�V����o)�gy����GU��H0�q�5d�O��0��sw�iAg��ywj)>])��J���z���p�= ���Ԣ��\�C���a��1��HN�*O?
@v^�:�MWml<��X�v)I_<ݻJ�o��c�?nK9�i(���y{�]R�k
��C��4�?X<-;��������|ꂷ�(j>�і�`s����"�+���
{|}C�+1���+��>|9�Q��,E���&	����3�!��d��=n	o܎���3�!b�P�P��q7�����='��S�>t���ψ�o��1��GA�o���t��M{jw� O��n��f�܂g�Ms*�N���&OMx�d��;<;@����;57�<�8izCJA�β��#�omLb2��[�;�ӝ�7�ܦ.X9�N����k� R����*w��@��9`� ��]�0���o:�Ï��1j�:m�e8B!�G�]�$���9�ґ�n<~e.�0�0F��.!Q�p�k��5 �f��� +'#�T�33JVI��j��`�����DƱ�u��CD_�0���1��t��#+�ςQ5%��z����z��gde˞���`��X�k�Rqn�I�e�r=,�!i�������d��|�����
�	7<,���
9�M}~+�u/д?8QCvz�鉖�~��'�*��r�/,ʁ2K�-]	�C�ښ�2Fo1��R�v�V~e���`��j�΀�U<��$g��<lB�#�dw�?jg|U+�a�7������:�w	�nsc�BK��rz�?YahJ�i�ݝ��X�T!�(�Λ�/�s�@���P	?�j�M<U�_����K���虝aVxv/GO�����?�7Ԃ�Y������.-�|Fv�P�ҝ6t���&��&t����F�R�SQF��]H�u�e·����U���e^� ���sZ�{���3܈m���`��t�'�� x<x5(�+��d����T�6�Qeg���d���/���P��f��,�� ��Z�Cz�H]v�G.�aHYAFQ+������d�ehXvن?g� ���X���)�����+��a]�̤�$���0�H�1�������s�E���^c����U��gA���m��is�w���v3��
��`;�~'+2/#�����Q�qR�I�м��s.���Iɼg�����������Y�գ8�+?�5�d�������5{�ܼ=isbK��׆W�Nܔ_<�8Ϡ���GEW�oO��v縍�L��Cމ�I]�+G�����gUax��#�E�ӴK�����iTQدӦ*�l�}�b�@����r�;oF�W��_�jI�e�T�R�\'I��%�e�Y���K/؏m�3��tx�+� |����Л����{�������G�s����< �u\���#��ᗏϷx�}���Y����K��G��8�e�{8�7���Ī&�A�X�&��,�;x�)+�x���s\u�ŗ���ӈ�*$�xa�ˆ�+�N�R�U��k�Zm1d?I=Xny�j��a7�Ul�- ʼ5�K��&�ks�^�d�2܎�u���i:��D(c_�a���[kt��z[x���o��=�e_�$��������g�@,�j� �Q ^�?�(��u�h�w��(�G ���ٝ��3��G 8�Η>Av�c�#m>�LF�n���ae���ͽ����up�r�~V�MMQ4_"��-{�o�?W�
�|�?k8�?Z�Q�ݣ��4�������b�A�<��ev���q����@�%�]���������L�=fX1��<x��9����%�>������PiA�` ˩�D�B�9�a�c���ʯ��o��I�����u�%g���3�ZQ��/��.�I��-3���Q&r�X�#5X<����D��A�
^��r��n֖��l)][�U�\����4�휯��l���p��������~���F 
�z���tGE">n���u+�F;c}��;^`��*�:k�On�_5�ͦ��t��ף%
��ӈ�-��%٨��oʶ)�O����m�n�F�Y5<�~�r/�m"�Q��M#��֫�� =���dBI_���+������E2Gn�t�@�d�(�;��bqג����3�E��xB�#K���J��1o�+�5��}���]ρ7wM��|�Q�D�6>����y��$�>_[~��D����&�=Z[Θ���'���Ua.������s1:�OhY��*�*?,�O�Z%��Y���|y�':=��u����OC��'Z�S=��^zw$��9�ֈMKǫ��;iPt��ϴg�c�Ӵ���H��S8�߿����Y���j�L���z��d�2m_H���o.>q�!ݝ�����9��\��r�Ԛ������t����L�d��^-�6����K"��I>n��C5��df@1kJ"��12��+HsnVA(�i�R��8�:�8;K2��'����,�=WҊ^�8�B��2����ZSI��u���!W]�݃e,O��V+~�+(,�Y�	 L�0���|�A�	J�qj��+M+������l\K*�9*�͸�/�g4L<�	����LIJ�=�jv����@/T$f���7%ʆ�2�<:vv�k�B ��kfZ�m��:I��T�Σ��^۪��JB$���q�%Ք�1��7�;<{Ȣ/���8�&I�t�lN��T�����)q�$$�-��X}�dt�y�Ɓ�%$%�"G�z����]g�% ������/�'&+��!6U�'A��E+�h��W����Xk�`��2�U6:�ul�(�"8��H%���Wxp�3��� �{ud��j���Hx�����"�dP`�/S���'�;�{����|��)���^:�뻣�	�"D�l0��ƫ |T���5)���1]�x�vIQ�q�3���}�\�����ʮ�œR���
��4d�hP�F�� ӷ�2ڛ��L�Qx�C�M��^��)�+�PtJ=��//k~Ҕ���h~t�*���+[��Z��v�r���⪥�b�M���;=Ta�,�7��=m�z�S]�2?��K\�Л��g�Y&�hK].��ib���RE%0n�VNjz[�zh� ��U}C�g�H��A�Rs>�ך�{��/�k�W�cg���sn�o�ד�u-�"����ϕ1]}l|�߇���_�Oir2e-�C0���� ����5G��\2��0�$��?�F�L3�0p������
vG�������|�9x��_gȄ�DUjiR�Ԟ�,k���wb������O\z��T,��k�"_�����g+]�OP%Ǡ��TÜPr���i#.4-�.�#��Ux��4�I��xނ�NG+��8���IӐl�2F�*�MQ��⩻�<�^0�gњ�t)��V�}�,u�G ������J�,Y��N���e�L��L���W���?UE巚�A��K�v)&�C��ǞF重59]�6p:]�QVn�����l�-���o�� ��`BB�#k�©w.Җ��é4����s�rA��� ����cf��k(B�d
m��G8�9*��>������=COKF���6<�Ⱦ��������ќ�0���P�7�0w�c�l�ٙ�0?EOO�Nx13�����0C(@�0��%�r�<k�ѸǕ�{�����bWkw�p{j�:����]w�m�nk)�S����w�ԉ�ϗp��u�=6�_qa�w��y�®��n
�D}����>&�ltƐr�0�oF���,�t.w���K�:�^6����7�Qɥ{���/�uNng��Ҩл��8Y�ᑙU��h��f���<�W��֋/S��6�y��z�	��5U�|Rx��[�O/�%���a Pf�?���2'����?6�,U
/��~��J�|��7ʻ�"yy�]���; �=�#���d���A��Qv6�5$)�Q��ӻ\ P�?�Ȩ3���mcW1�����,5B����4||�o �����ǼHS���}elj����k�-^"z;�d`�E<�YI���)&���,U��>�7h�풜�Xh, p#;ZL�\ �p`��~4M��S؛]����-�A��q|�c�|�*��A�v8HK����C���O�������x<sts)���xT��n$-D��>����~�<J���4�,�A��Q�Ԑ��0P�
�Y��ݏ-�P<����Ϡy���+��mϣ�ju��P��̅��\Τ�kh��X�R_(�Я�j��8zt2����|�Ь��u�Ti�A+����/��� �g��0�`Ē��o�8�Z��"�CY����etⰳmJf�L��V�%����X��~s?�J��K;�i�J���#)M�k�[I�%Ow�f4�n���,%ϟ�v";ܬ]��Q�1��߂N��o�^���ś�gf�N�r4*�%gꍯ��k��H���;k^�ؐU	]ov4\��s���\q���\�uv��/�W�_;��-�ғA�G>�|�w���g�L=�N5�殝���Nb���o]�>I�O��~Z���.����q#(ɒE��/��4�{+W��8ZX�*�O��&试x{�X�Dd����!�g�?���O1瑾�=v
�܈B���'��Sr���Q9ݒ��>�y�e(�PCrӾV�Ј?צ�C����?�&>�0ue��/�����R �e��[����4�k���%�XZ��nkv=fOA7��Tt6u#F����n����d���0��ʤ7e�'U��J�؍/U�m&a���O�g��4����VC2tg�K&����ܣ���(_��Z�>#z�#�؏#td�bT��y-����t�($��`���W�@Y��@�ٌ�Äƕ!)㇯���5A�v�^(�u�@�+�Q9@-
���d>E�S��g�?����rX���{��q���H��¡L.�E���U87]�&~�/��&
Xp��k�������':���0@��G.���A�,~���n(L�Y+� ����q��D�^¤����(
 ]��,����I��Q@y��	,�OI�0t.�����ى'�Iư�"}8F���`]KR�]v/ܸÆ6��"� �F.�V��0d���=�D{p��b�;�  U�8��y*�V�P�5�K�ț�h�kc2)+��cҷ@�����]������*&t�ӚaJ�)����{q)�!��`<������=���BI�'���:�sAa���tS ௢+Aѻ�����p| �q���]q	��@�*�P~!i�0� ���u��o���&5��e#"�F�C���4�A��[FDeY�t�D�"�M)E��L0*�AJ*[�?��jOB��U,�>���&�ޙ9������+��/zmYaǚ�|��dO��B����w�r�O:������Mt��?���==[{�6)�|��T�0|���D�j;;;�e�,7�8������5���J�O�A��o�(����K��AX9�{����&2��;<��2!(C&���J~	ᡗ��T+��fØ�_6�������qgG������E�ʎ'Ï�Q��e�����S<�q��4S�X�NM�G �|���]�")A�g/7/Q�*��S҄9����@I�|�Z�c�я�J�Gm�i��6#�1I���m���Np�YZ�%��]�:�����Ho���o��V�?�P���k�`ڛyG�����Dbey��)��j�Y�/��y��wa;�e}�PG"+�I�������L;*>�{�K�ɍ��v�d��[>��|����-�^,\����$4��ŋ>�-���~�,x�2�F.��.cf�R�R�h����^�;���A݋� D��Rbh�ș��t�R�v��G/�;���]��� å���9��h⊆����&t%�5gaw�{�n��g~Z���Cr�C��D�F�-�@���7���T�o�,�87(0�����vK&�f�/�"�vA?<�@t��� �(Z�U΁1Fw�����&p5=o�� TD��E*RIe�RD"TZT�T�$K�$Q!�IJ%K�]�
i���hQ�$%D�����gι����}������=��3�̜{��Y���NIvW#���!��
��Ǉ}�J�q�	ZF\��H�Sy5q�?�iB�ɾ`�p�����j��W�)�WJH[pꢭ^�|�VU���.�.K�^��Z:���-��S*o��4ʽ����TOOqݬyi�l>c;=�}=u�f���EV�^6|�m�[@�^P��Ǖ[ٻ��U>�;�,ݑf��^�aA�Ȏk��o�O^��P�ơ�M��G�m|s���lsT�̶~�;�Ϋ�0>f&�#zu����>�ݭ}�����c{�n\76�z���i���Y_�X���]e�E��')ޑY��$������r����y{��ؐ�{v��h�)����o�r?i>0���J���]M�8yǩ�mʼ%s���M��R����̠�������쒈��q) ���Pv�t.R�����U�ʹ�dk�i,������)�m�Ń�]=N젍b�w1%�#���]�"�����S�,Z�D�[��I"s��	�@)�ѫϋ͗�=V�-x<U���ZF�9?����N�]��>L�|��؋	Y�۵������a����^�
n�}5>�?�H�p��Jmy�v +�^�����ʡ(<|·?���X��x����V�DݥsJ����_�9�SN�?�� �[�߰#�<|�Bر��������_bL����w��Os��#�DCC���ı�į�!4Fx������|�6��c���Ǻ
��F��:�КR�L��CCC:8�e��ғ����V���l4�'4 �v�d�HV`��8�U�����e�K[���K�y� <�˱�y�u�81�	M��[U�"���T���s��o��~x��?0�n�Sl���ڐ�޸�6�F���ӥy��zSav�;���C�@��l�s`Żر���*8�πx�FT:Ô�[P9���b�cN�q~ZI�q)�i%+�w�@h6=؍=1X���z��I��Tr��.�DH��r�݄б�-�������5J��C49pK��qqǞ�P�2���W"�nD�pi3Z�L'DQ��[w�K�����A��|(ձ��k�o����v���$߶{�$�Ք1��B!�-f2_�I^�A�:�I��k7ӣ#<�Krx��޾�H�ZR�0���5��{^��"���>��u��5l�]z|g�^M�,E/��#a؛�����W�v�ƍ�Z�V<n�f?�Ŧ#�bR�m�� ��b�f�DC�KW�T�yp);J��H�8���.�:�0�E��k����%������k;�Ɏ�C����$�)�K�gs' �٬`B���2ٴ��l����k��0e����;}QxOw�$L�q0i�.�#���ww��m�d�.-p�.yA+8�#Ρ��L���B�6�:6Y[�0�����Z&eE��\|�>���{C;_���{1��[��h..��ᒤwX���#m��&C9*����{=���,�r9)���\"��s��O�{�?_kS�*)��B2Rk~գ��I<o�C���h=��u��£X��J�9�Aqp�aX�t!�.����D�%��Nd硊�(��u�����J#>,V�ePzC:���wR!`B�X�]a��k��5=�ݵ:�S=��w�����^�tl��G�3c�S�9����e��넒(��o�.p�L�Qkl\�-�퓝־Io��Q�%6���������~~T"�6<o��;dC�%���{�'g)�5~���΂��s�+?�9P˳�����ʖ'$������zosv�����ƃ�Q����[�����iz�������b�vw(�X�	��,��Upt�g�xY����Ǘ�:��Ȳ�!xȅ�ZEZ�{�W���!ep��9߾aO���iʴ��j�<$�<���'N5�!>p ��J��.�'g'0���`�{�s�C�lpI�8�}�R�?@��[�߈���<����	���:iӑ�	��HD�_����=�#U����	wz��?
�p����9>y���ՄR��9���!�p�����8�ygq�x֬�g��@�7W�����t�#9D|� ;R�h|�TT��d/��d�4���?(�_e�u$%�=2b��qb��A�;dPh\���,_��c��ԑ<Z�wn��b��m��fv�.+�O�x-~j�h�c�9.�8Im]��\�47z]��q��%�,k%�^EN�v���ݣ�r���ڂCFO���zd�h���f�����ۣ����*gb['_�<��g���
����U���l����T�N�-��w�w��[ƈ%��Xv5)�tEC�:"!��e�g�4����H-^5geY˗��#�]�+����<�u�'��m�{]��N:3���ō��u�
sӖ��f���E��T?��[KF|�_E�DuN�u���5~-[��������Y�oH�hn��=�;�ڎ���ӂ}�N4R7<�����z��΍a9^��mU�o�)������`�x~}�H�����Y�Y��*`7k >��7� ���� [v��A��!��:��L�E����#ګ�d̟^*�<�'L�{b�Cbↄ�c��g�	�m�����]���!111hk���i�@�O|�E"I�w��!���B�l�Cq�yM!B�6�	�/5�\��<o�4dO�:�'plz�H���U�ם���c
&s���.���Y[�m>*P_��iy�],1��5�]���[&���ԛ8����d��ն[�+g~���8`���+)�G�:�N�\� V]=���ۅI�b��l�U���<h���$�e���m�����7�*V�nͦ�v�smgZ|l����¯,�⺮�ܐ�//�0�M��8-w�g�\�/�(�w?��w'�V�+]���Mg������sgmr�ڧ;*��|��a��8��ΊWJTꞭbt����%��������:~���v�\|0���k�����x��ܶ���<�y�b��}�N�.���ɅJٙ�%[,�7��E�M��l3l��-�
!#��2.�;ZI4n���ў^O�w����,Л=eJō0��}캏�p~l��Ou���X�1}M�K<�щ֭��>6���I��lx�x���~N��:�Y���"�;>e.�$�_8��$eI�]��e
a��R���iw���ڲ��U>j>��xm����=^~�ε�3Ҕ�(>�?Y.�s���wC^*�,I�ܼ5k�����}B��BV���n��Xn�.�6�e����"�uԷ*<�����^�]�;{W�+k�T���	�h$End�Qt�S%:�{$,���u�_Ҽv�9�\D�-ٻ<��F�"4�l��Bj��;k�Iqj�d��7'���%��� w�r]�����x��~S�s�7D���Nf*1R~������9��&���g77��|���pȝ�D���QP"�_�F����U��/FC_HT������g��g-�F��Y%���%6��R���n�s�O�(�W72�&Q�*�:t�e��|u�}�reM�}�ɠ�0%�y�����^��yk�~�<���јТ��wn�3��k�xu[�x�������.&o�j�8t^f�v��7�Cޣ��wѣ����3e��z9�=�\2�>��i��my�-b���q�w��<���-�#��KB���sC�X��O�Ϟ�T/8���h��
�b�u��희��8�2�h��q�VK\�x���Y۸|����R9/؞v��櫔<��]�t&_����`(�m�G�*�($� 3��Kk�?i���db���3ݥ�mVm�!��c�7�d ���|����&��N��J�:���fv����P���~��b�����v̉���h�u�R�u��8�]K��qh�a�9&�+Te��,S����7Q�SR��*�\^���գ��������Z!{�J*�`�����0��<6�g�_���5?��JB��Q�\�����:���.��X6U��a/=�bD�N�B�<�� ��WL�}{�0��j�~'���wN��)Vg��ӟ�yULx�b��km���}�l+$����q)Z$y�hGD���)��?<�:��C�@�t�)w�z%�?�S<$�r�Mey�/u}mf���u���x܏/(�j/������}��3����RQQ��쿿�,�n]�BY�IEe��y��sm���P˕��_�dW�9m�����q�����>��Ŏ4!�2i\��/\���/sU��@���Fh7�o7(�+HAE�
�T��ʲbeb���]�����g�Α���Dű�}��C�Ϝ!�Ts^J*'���ς�����kS�Ӑ�G<T��ʑ=��<��X�~`�x������H��i3Iy�^�t� �/T_�#Ѧ�;AN�R¼䅆E�zM.��G'����!���] v=	���t���?p�\��.�-���C��.)w�΅��0� ށ=�WG���Zk6�A�!���@�#�p����9���]Z�vz��<Gy�����Ě5k�m���K��2g��d�]��&�Ic��4}A���L�)��xvr�G�`W~�Oų�A�Ǟ����A�ȕx�(��&�J�6�~��=��c��~��`��s�*��s%�`+�.��#�R��h�~R^3 �=уV[p�R��v��[�VTIwa94j�_��"I�P�*\p��/�Ў�s�rR6,ǎ��O��p+�>��cG�{8w����.6�;�͵\,������B�C-�x�B�93� c2��?{��8�9��e����P҃��Y��t�t!�t^Zö�k�s�{�4x�Ů�����gY8!rYr�#�뚙�l��pڎ�<3�HII	�-���E���ʔ��DU=R�p[��0aI>���i�'�K��T��_�q��^VT^:���s�C������Gs��	�v�P���tYs-)�>��f}�L��{�t��SS�_R�i���^�5��l�R�ՇKE����?�jyb����Q-�׻_��Om�BA���Ǎ�%�=��J̬��/�_qʇ뛥.�?.��_�]�ۀ�^���B2���8��E���5�o��YǍ�G�}h�|�0(^��В���m+-��-���y �d��9�������7r��B����jX��u}ߥ�>��_N���`�/nV�cj:�p�R�'&&{��^"�SHv��`b�����u����/�)�f#�6�=�s����n� G��� �Ms�{g��1�r���*�m��윌@�p	�AԮ��#��)Q�;�Z~vU�j���Y+�%�m��C���[��Ҟ��w~ ��S�3ĽSB}vI����I ��l#�~/&�> �eCt���4�Kz�~WS1�Ю��
���`������s�\R�85A���xH�\��3���ޱ�\`��].�������C��j[h�O����]�a��F�K�S�W�~��E�_����n�����=���cb��H�V:dj�s8�cUw`�\�k�Wu�d�<��8f:�J��p��������#�
��&-�C`���p|8�ox��@���Px_�>���Ԍ���qP���{ �1A������}�W�|��O����F�1�1�`?��/�����'|.�	�(⅙ ��H�?�㤺��{)GJ��C_[-�w����)�${Ɯ+������3�˝&�E辿�wy��F(��{��;v���_�H~�{y�¶�W&�+�g�P#B�aS�s��n��	�#��Ǵ���x�� .�_�r� �� ��w�P&�@�����U��x(-�A����5�~D_�KƜ�&o7��jX_��ҥ�'�깱���c��MV��iI�)��!Df���4F-�d�s?3��$�!H��Ո[cO��k��pO�h[&ܰj�˲j˚ܦq|�?��-[�n��,_�I(��^�fE��dk��nj����4�p9R�і����;*�A�E��wx�*?�3�x��׸�"!�E%�v�Ue��U��>[�H=���E^�@E�����*o����i$��1ȷh5�򩹤~���ڶ��C���)h�[�/���s�'Q�7����?�j��.Nkf�"���js>���{k������9g<v�M������|��%'m��V�2@�#�P�#�O����4�[�3���1��@=*Q��OE�Qő6d�`Lp�\8GYqW�R����ʀ��t\gO�R�����f:��m����Z�[��u�6���Xl�0�>�5���_SY/!3F��	y�o|�>���b�1�AIմ��w��N�&��""�P�����#�Y{�g�%
Eb��&1�|���	���Ӧ�S��A�ql�Ӑ�O��L3�c������C�y�io<F}�M2`����e��r_G��19�*�9�r���J֔��b��[���?�;���uQq�)���=��<	̰|8���"myG�7�Lw�Vmv��2��yΖ,il������`f��5[���=g�vj�_e�k�#�1	q����~�O���J��C#�^�ˢ��nUw�4�u�7!���\�``nϥ3V�/�+A]�	��VQ��"��"�����C���/�#Ў�S����1f��#�_eo}����k���oR�>o������G�j����U���n�O�ǯH��D���-WߔE\9-?4h��ꐶ)�no�'v�	4��_��5���~���p0�������[�z�n�x����;L_��XU�����q���tG�~�]G���yd�+���!*v����~��4�����k%��b�<+֙�,eϢTT-�(�~/�#��\�0�rHGO�jk�)��'�P�,�6:KU��Ox���S��拢���7�>����OУV�� �*�4���*�Y2P~-f� j��_�y�T^�����d)Xy���6Fb;�rP�z(n��jhh��7>�$n5�Diy|�s�؉��Ј!^4�?��З�b�����;��~a)aks�S�7d0:D/TW��6k�	��+�����һ�`ǂ������Ґ϶���R�ݜ#�A�j��ɺH@��dE�_��⥒x�}Ոo�4z$���qXAʲ����Uf�W�Sr50B��#�]�n�'��ݶ�
��:Ġ�U�c���Np^�^�A{b�iiH�R��N�����������r��.�-~�ޟ��' t��ǘqwfH](����9�5e˭�Wi��[W���.)��'�
�o6Y<낦O�g�]��S�S�8�eru�H-����i=W2c���?�׌�V�3֔3���C�|��w�%T�l�
tl��c=;˘���s'��R���B`DX��J_�:wS1��?# ��&����r����$>��h���(dSW�������#C�Z�ז�6H�_��s�tEг�ӥ�Nw�z���̢B���ߩ��=fpgߖ��W<�m��6�*u�V�1%A���<R3�)�s2o����%<\Җ��P�+vC����s�٤���jW��t�M���1�gF������w.33�xx׊o�����
<P~��y-BY��^7�v�í�+�[��Ly����>uv.��yfߖ�s�so�E�ׅ�l(6{zf���H�D{�i_O�	zl%�B�(�v�ݲY+�x�8r+M{�2���x�]�kì"�U�*j܅�`\*����<Dm������{�W�E���ݟ]:{^�P��!���m%[��E�lYxсp�c<�����O��9�P!�I\����d��B�_5'��ci=��Ma�8�@nL��8@­�o؎��z��z�p|*?lSq�+��6��(Bf\ڿ�.����"��o�(�U �#��ޡ�kx%� e�3VkL#��p{����}o���s@J�����k�dY�;!�[�U�2��� g\��ӿ�;A���2�I��];��,��M#���u�J�ܾ��N�T��(/?��!_yz,�p���M�����+C��&>��n�ɺ��|ֺ�H*��43el�BV��ʓ��Z�U)z>����O�ni�w}Ȝ6��U��m1��'X68�v�<�}�����]k�g|����3�O�
ī<��笍�<Ps��E�O]+>�~�۽y�}��E�wC�{�F�x<��O��ӯ_h���>\���~����m�W����[�술�����Q�\�����>����+Ϝe��ޯ�9�K���)$�l���a�8��r�qGz�}�:d���OZ��v�9$ɏ��>�+����,�i��_�x����~�W��W�s`������{,9�AH�PЏ�<qצ@pxNmW�~	�W,}�����wk�{q2�?lw� �UG�!r�t�~t��w��x�3eU�;)KU�`�:��Nxdu�AJ�&�0�o������W�4J����`�}-���I��k,�_�E5A��Q�<GOL��
xE���"�LK���m��G�{ט���H�a���y7������7���c�C�x>��+��u�)�Ԭ��=p�E��@*��Q��	c��9d��a=A����X�]����0p�8�d��+��]�Hg��ݷ�Y� �yC(���-|�����yp"�U�!&4��5x8 �u�^�]�[H;^�]`]��p�K�q�*��[�3�6��/�9
������#�����v�_d)2��uA�+N��A�[�	�w���>��>:��|����1g1�r��#�,.>4Y0��ٖq�&� ��z�;��W�b%��C=�,ٟ{NJ.�!�ˎ��O�����%���F�Q^0Ơ���%�I|��C���ҕ�7)�:�>/Љ}����#��6~3��W�����R��Z���K_�	i��Iy��+���,	��� ���*�\�$mJ���A }I3�;H6ߩ��(���Tz�M�+��	��b�Β
\'������)�.����|L����Z�G��$��Q�q(�58��m��-�|-:���A��9׮]Î��;�_SL�����	�oZH���]&o��lp:%��LlJgd���4d�
C��ʑ�ie�S�0J�0%iN=�DX|;�qx��aSِ{լB��������������q��w��{̄W6YK��M�.�s�Qhl�ee�wˆu�x�#J��aP�0$��}�ua9;ʁ�s#wq�B���^])٘��l�M��Mғ-��m��������/����2��wa�����#�`p(��-A^g�
�bz����tk��O�	�^������l���[/�hdO�������"���ק-��h+�2eò����}���.5ezcC�W��aC=�/��d����B��E��>U��!>m}��~�7i��0��[�s#y�um
��eg�UK6*�+:8<\���[>B�.���ծL+^8l�I�X�V�;�L��d��oǼ�C>5:�(zͻo<��:>��([d��W�6UÖ�,s��5|�ڗ�&?�E֎���t���u^G�&�v]�d<��Y�b�]nF��˙�E<܈�n�s�mÛ�0�T\�dCS��}V'�*�4�Ch,�S�&Bƚ#�H�d\6�韺6fH%W9B�~�J��Ş,�n�0i�����V2�YaK���!Y&�k�*W��5��(	wl6D!��Ϝ=(���@�_�S��!G����Hrs6#r}�#%�����#�,H�'3�
��=����/��|���0!�>�9� ����{��\������hf�4���n���m �,wd]دf�O����S�-D/Jk��S��C�u�~�zֺDv��3��R�Α�ثq�Au��13tӓ���?ڰ���ԕFS�h��XS���]�w���V���t��ʐ�SJK�d��|���%̷-q�j��owN��i$��ZF����M/�0�\j������Cޘ��nW�H��g�?l�>�+*خ�Ԉ��6����
s���N��O��yK'�y�e�MĐY֪Yw�����|�r�D޾�NC�H��j�	?�yn[�}޹�9ޕ�_�j���z���^���M�`�/9�d�{.ǭ���g���V����|��WЭ}vf:��?.8-g� �Ӱ�,He�D���� �4dXЅ�˦s�:v�����TΥ%�X�Nc�Pvp�]`���	E��q����.�E$td*"�	��cȀc� Y�:v����
�1��Kmo����9��x���#�q�T��@�AK@"WG�bAfY�"�t������^�w��DKp�֓O��.�neb �� �I����^+ӤU�����4��Yj���W �D9Hwv��'��=����I>=�C�lm�;���pbR�G .pxH6�ss�K�"7�?e�E-鮘�h�SI���yI)�F�, �ُ��yyvyv���F^����dP;�U�	��!�w!h�p�����*�GA��k@{���d
e\T4!��,cc��Ըݤol%��W�'�J$.�ݚ*�.���JE �`��<M��^ZD�]*���߉u�#g�+ĉq��.�e�𮜈�&���ϭ�(�a�������3�t肆X&Э��L� ē����B��j�p�y���d���y33�Cu�3���gr۱Y�%�UpB&(3�J'_t�-XYi��&l�6Ct8?�$��IZ�J���B�L��,��3�[-nhpAO�e�������**0��Ko9�Pb�V�,��#t;��.$�q�
�l �����r�����"Qg��L-�o�`��&�����m�w�cST@��N�1����W�ޞ^?�1#����󗘉�úi#��݀>�����/�Î9��1�.sƞm�V&XgL<=�3�_d8��1�yO{t��#l�s��	3��X5)�P��S���!�!/�.dTbc���g<aL��� �A��%�ag�@��~��,��?�K׺�+j��B�#���7�:�*�K�"V�� �EPߴVxw�R�wG)�zV*s��ǝO7�"�^\	�'gl1�1����*B'H���+�V�A�������MC�j�(5u[*��l(��Cd�n�����ו�׊��R���
NT��P�P��Ǟ�5�Τ����������9!��9�6-K��>������7�י�����/��	��v��q�c�[>q)�sC*�Syo4����ۮŬ�P��+�q�e���{��@J��:�w+t9v���"��|B��~<փ��{�:>(���A_�X��J�9�Aqp�ap"M��K%c� �w�4��y�"8JaD~]8n��p��CV�C����]���B��ұ��B�7 �[�'�G]��J��C�zq��Ln	Wl:8�ռ[�n�M�C�C*�:��kv3��3�����J�S���M3Q���cq��97tO�N�6h�t�n�;g+ܐ
�~d�@]�Τ�Q�W�_&qNP���;.�Ӥ/��-������������&-���h��@����/�&;�u<�n����o��y{�W�ui�z��D]y��<���~Ȟ��p�.��ݞ)��P�a���=j+U��K��^��]Gz��K.���o/�@�x�U7���1���|���5�䛇� �P��8�	� �c%�dB���� ��W�V��\�Ґ���u�(�y��V��h#�V�7n���V�E�*���u��?�?G���?:�'���?X�Gp�T0A�'�97lO���^�C~ϑ����e���
4D�m�>�
!2G�� ��`��㋊M7�伤����Hx:���ǩ��ઊ�a�=x!��Wc˷�RӚ�k�cIM��c+����p�ᬊf�Z�߂#��5��$-�O"����?�=�ě9'/0˒�iy��j��BЧ�����:��d�'�#�]�^tQHN�^��^�>��"[w���T4�6*��*�ߠ��K��IW���������������9Ymu��ͥ?�R{�+�d)jj��i��B��Y���'�4��7p��k5_��x=�Ɣ���>j~]��o��]�v|l���c����%ݶAfF��u�O-2{�ĶVI}8hy���ћJ'e-��v�����`��$��۳��F̑\�X�i��".W�^�����i���膻ȗuq�CUx+������~X怾o�3�c�`�!K����nx��;b?v���1p%xV�f+�k�OU�c@�=�o���q߁���E�iiTt�]�)�ɦ2��p��u�Az�� ȿ��]��ʦ�Tp������'pA��-��]��0�'�=���Tϻ�����S�7I�@�����\���{�f��Ӽ����y�����L�T�z�M�8)˻��Zs�E�k���Ռ]����Q��4h�+�������ٽ�֍�<U,����*��U��}�Wu9jb1��m[����-�ϐ��:y{�&{�W�ݚ�uN̸ś冕���{�q�Ty� �ʠ7�3?+6[֬W�?�`�y�9&銊.[/��\�i��q���|��k�GVϟ��஋�R:b�j��j�Y
�y-}y3u�Z	��1ۊ+�8����� ���n��l�����ܭ^6��mF�%��[��E��
���&��q�����(5<��-E'�pwUū���n�_��Y;��3��ߴ������}�l�v�)�{ƾ}4c��0����=ܶ������w�_6oۉ�.�	��sր����Bvo�X�:�Y�z����r�l	mϭ�t��y,ظ�)T������&��*��#�L/��r��s�9��w�T�>�꾱�U���v����%g�	%_msjLӸ����>���a���[��7��{���{ަ�*��}_�6TO�{6����o�6�X�9��fmؾ]3��zn^���w�n/3��6�v�og.��!t���Sz�j�Lv�o�Jz�8�"���muqgy!�>~��	�uW�yo0j��)�+�N�VW~�����Y=�#΂��j���j�ϝ¦s��S|t$��!:S~�z��AH�yx�#���=�(Ky"�#�E�]O]Bڢ����U��5���
aX@�B�Q�ᷧ9p�����.�h/�~���mf�AA�~1��-�^d��Ƿ��9����BBQBB��.n�� mW�($�JĜ��������'m���Sg�� L�T_5%�O�6�](핼��a����S|<��{�kƠ����-��e��=�r���ׯ6�2f��{7����i��>A螨�������ӽ��<Q9��}��ܦ+���|��<�����-O��+�6(�����,�b�y�j�C����-}�.y�s^z^����-lj���"yd��;�[�בb++$��4��=�i�^���{o{㶭�>�MA��r�2f<�Q9�l^����ѧ>�L���r�`'����E�e�}�������1�ӣ�c\x�E����܉''O^n4+W�f�9����kT� ���&G,gf"1�v��}2qr�E�s�Bs���:p0�v�������MȞY�PC4��]�b��:�t�N��;�I��Pi푽��T��bl~e��/�������7
r�T��Z��s���m�&�t4���!,��Ţ��O�fY٤$�G�;�;���?i�k��Q0i�٨�)��=t��&SutT�S��G;��Q��o��2�/癢���ތHL���2F�iɾ7���F�;02`����;���&����/�y�����/ ,��#=�t��ə_g�
398/fҠ��F2v*]\�%�z��!���-K�e�I<m@O�����u�z���Ŷ���W��axw]p����)��5���Zh5�\Q󸞰AR����}L��eL1vmRR����.���9=/��D𦤤gwCF�f�\�҆L�[���!{Ҭ�c�cq���Τ�:U?���漡�U����ػ0��љ�H`�!@AFXA
*V�"�T�++���('��?�u��l�$*����Ϡ"~�A�reQJ*'���ς����ċ5@�ۉ�0���ًXk����E�Tv�O8����l2L�jk����#�dU<�����}h�y��������u"�}5��	�_ڞ�·�g�qc��.W\� :��v!0ӑ5^*��2.X���>�3c���ii�Es�)R��t�z��]����;��� �%��]s�-8�5�Si��?����s��#��<�en���:�s �׼9K{�#8?qr�ϞsP����!^Q���`)e<��b�!7@F���@�:�=��e�S�_��\�0�����8��� ��.|P��x�$���ċ�O�q*�o����CH����ǥ�����
2xٌ��va@k�
����z��C���/���z�RH-?v��?k����au����A@���o��౺��L��WE�w+y��zN�&^y����9L~72��2¢K�1�A���\�n]�����T4�Kv^dҕR��yi����Ӑy���q�Sxq�~�ˏ���c�h%�T�C�Q�g������y/���ȼ���;���yDU��l�-xDX�������SH|�ү���R�����t+ȥg�R�gƙ�MҘ����}.������1K�&%O��!�?J���6���[�T����;����(�F@��@P�O�m����\����&!6qc����s�U4���st��XT$����>�U����z���ǡ/
3+|"'�pjZ��t�䳉��^λ2K^l픫�3;�u�'��{��y1`��������L�tܹ���{�)ܬ�x�󗝁�9_iЂV|4���,:'���~����/�m�)%���C
Y�3��,��_<�PȊ�=qٻ����t���pZP��KX�݂x��T���ӕE#A�t���	WC���~0��ޤ��x�������x]U��حlJ;=�{�/��|�+s���x��1�KJ؁��Hʬ��`?J�%|)i�5�n�|����������"����Q(_w�N���4xm����Q���C�����V��bwё�x|���xЏ����[A��jV���*��헐�2��cU��]���С2[S�˹����-���aߛ��C\r�>��� e�<\<�b,P�*J"������7��cŽ@��C�rD��#��|s�zC����&�0��g��*��O��'2�� �@:n ������oAn�h �&Հ0���Q�9�}ʺ��I��2�_��� ���`�JqD'T8o��/�Y�#�G�^���Y�.��
���_�p�@�D�>�XԘ��{������|~>������G�����8R�&H�l�͡Č�mM ��q����.͍��̄�~>�eD#a��KBB�+(�}����v�'�`��#�������D#a=���( �⢜��vŲ�ۃP(xMwec5,�E��s��(�dj*~�uC*�M���TW����kn�wg�;�t'c0���+�`�[ҥɗ���($a�cF#�␁��/F
��o!23&٥�k�D�]�7J ��g����:~X�^8�����Q��٧�_}@{���q#W���*o��g��x1�����r��g��h�7�?M`P��4����݋�;\�W�J��,7����Aq1>3��'���:������3j�͘WS��q�HD��ؾ���,/X�w�Q��������F-K}Q'�.ph��^�g�^��-?<q�qE��a���f�L��zd���˰����L�f�ֻo����K�N5�x۾M�>�/[&5���t{���[��]��-D��� .KN>�j[�� �*F",���H�ߎ2ei,+��gı�+�FЋf|����UiC�
�K�@ l����n���2�,�ٓ������j���>�'lc7�:�j�o�cV��T�3�h;)t?�`j\E���ڛh2_�
`R��'fy�����0x�9��������l/��A��}��~w{z��1���:.\��/Z��8�k1����.3�"�.�x,��R�����ք�0�i�ý�^7�`ڷ��7//��f���}�}�\6?��<�՞ݳ�Fr�A�ѡyܩ���G�KF���-�ŷo���=��|L��a߾m��"�v����(�U�9��!߽ݎSk�}�)���ī/6�\��ݷo'`f��5[���=g�vj�_e�k�#�'��q����~�W��N��)��UφVK~|����Y	ZצZ�Y@T}��cp��zϖ�G�I.�Ƴ~�|%y��5�;B'�9�]����]�R�a7p;�~��УEj�Դ./�Q1�Se� q_q�sG�l��v�����-A���5�����o�	-�{0��=�S�qdǊ��/����������`?����+�Z�+��#!�����0}��՛�R�(��7������'N�|Z��xyK������o��Ô��^s�:�N�H�Ð���J��:�Z�(�z������3���[v5E_x�gڳ�'Lg������V���A5�ub��}������IZCϼo��Q����ϫ��[�v���z���wR��J��@X~l��@f�46T*&^t.�ox:r�$���b�"Y
V�c���yz�!�Mo�Gn�_bhh�g7�ك�'��SO�ٲ�%���������Y�I��R��*6[<�З5{7;�v���	C��b��� A��d�'� ���x��R�5��֯ ��5�T�I�|Ţ�a����O����J���ؓn�y�?���xC��ݛ_/���s`�d����7�F�4����V��� y":�P�O�)��;���������-�����yp^{�`�ᴀ>�ߊ���ΗTWWO���G����UP�?!��sڕ���qwfH��@Y+��$ؖ[���(%+�Æd����4�oI&��ΊPg��<��zv�LnI&��"���dx'z�z�؞�S����ܹ�k�Y+�kʙ���!V>z���Ts������zv�1����NfnC������`� ���N��n1OvL��u���'�?Y��ܪ bs���7�oCj����p�OSi�ӡ)3ڛ��T|v��o;����rj�Y������<�|��Qj���X̜�NqN�G~��{X��19�����I���F�<x!�M�Wbᓁe�"��[���gmF�aUy�� [>ǫ���ַ,�X�,�&�gҴ��]^�������?l�^Á˯�n�,��q�Ǯ�q��l.-�[�h|��O��^Ȝ�^t���������w��L�t�s?g�U���L����t��[�C����^�\�Xǭ�HL�ܗ�K#��@���ևx�q�XP��_�b�^"J�-���F�Aڮ�,!���	��J���5��L�v��4�\I|),�P��,X��a�떃�_�;f�_�7D���L�SD�=85�-��}{9|����bƩG��W�^��]n����w�L@=L�4�_��yM�3��:��HB%\ڿ	nm��we�5nj>� �)����L��](e�|�E �`}���֎o̪-[�q�co@�����a�=�}�lq;s�d�n�es�=E��5�,�bK����>�����6+������u~�XG+ׄ���7;�y<}
'��(�{��*��j
z�҇�sL��%�8�k������c/Vn�7y0�ҡ�p�t��Dszl:7�t����۝�Tm�r�V����j����^�_'����0�H�����g�v��S����X���E#�M=��m�����m��=���v^���˲�(��,�ʾ�c�&�k2H4g�D$�s���y����םu����nw����"w����6KɨR�oN��)�?]��v��M�31�t��N�x(��N_��Q�7��~?i��͑u����5۠�f����V�!�-+
�D|��U��W�] �_���N?����Qx��S}�?��K�;I!����������h�� �wb����0 -Q���q���ʹ5�
�Mr"I��ٍZR�9���:<���z&9˃��n����V7�re�G���o��O�U�l���ι����c�DT��W u�
_�<mtYs�_���o�����a������,0���h�HZ�[���k<\�Kʹ����2��>xR�u<�����h�ax�I��B�'����@����G��}Ĩ}P��S
إ:� �i�	r��c0�W�e�E>��#���ћQ�@��֙��z8�Lϋ��2U��ɝ*��s��w+�É�yq��x�`�_�I�c�����Llǩ�	��p1���JfFXp�O�ͭ�<~�/�9
������#�����vx�8L����f W�����=9p;r;mǕ�tx
{�ޔ<�&�ztA*��;&6Z��geN�W���/!���CknĲ��A^wW�9���G�96������a&��V���"o�h;�2���I�ϧ/�TE0�\N��T��=R�P��њ�x�h{H��1��߃|ߚ%�yY9Ԟ%-�iB+O���B��������/�C­�?���à*�\P�8?��d��TO����x�J��@?���Q�����sLJ���J�ȧ��bj��*�ī#��@���{!X?M��d�5��S��G���!7jr��s$�C�}�"m(�D@���ÇC����Ხ#}�;�ۂ7=IS�;��X�N���	��E��,�+ ��T�\MӞ*�Y�XF}L=�DX1u�x!�լ8��}�����CX��j������9��B�v̈́f��|��O��5q��	�w���kCs���f�PM�n��e[�oV�"��
�ɳ�����Lp�j3���{��l_]Ho�\��m�������P� މ,�Dڈ����2!8	&����k���Ks�:Ŏs���X�_�"7�kg}�|��kPΩn����p@A�k��-]"/�{�>rVP�:D�ӽ�A�VNE%G^��WZM��tuق���5����P�b����ܷ�׃%�s�$�O�˜�kn�l�ɵ���Pϔ+Q�o߶�ٞI�O���Wnm���c��ӷ�|�p�9��C'g�MG�nS��t�{�5��Ϊ'�ʆ���n�4��K��bs��s��Nw�h�w�ޥH�u��e�����D����C�_���i
��2&ݹ�zS�<�qW��}5Y�� �yލNyp
�
��W��R��#�l��3��v����|(���~��<����(�!�I�Z$z�Y�*���D,Wm�qB(q�{p��%z�\I3V"wW�߇��e�l�_L��c��-�lVBh��ma�d���ɑDB4�נ+���$!� �HB�	����b��0�`�>O�+u�n�Q]��Y�x-`�}�&�m����iqr�P�Y�?A���>o�����G�pq�?����#�i�&v~?S�t�ë	��@���A�gr���O�ʋL��Zq���U��j���u۽pT�l����M��l\{1�1��t����fU�Ow���̬�ڌ	")�JLW=>��Y������~�/���_�����Ηv̺���L��RP��1�k�e��T�Xgzr��椧����@��qD��ƭ��M��^>��(���F�o\n��?h�b�w�WL[.zS�Z=��-�o�b�nt�^m���؈�:m����Y[g����˙���ڏ55����۰�|��Q7��Ψ�o�M?i>0���J���]M�8yǩ�mʼ%s��W��R����̠���'N�i�j��K"�SǞd�P�cXЅ�˦s�:v�����TΥ%�X�Nc�Pvp�]lȃgM�x z��K]��bnC!����
;Rp��00��!��9j���.�6�ك1�2�;�:��k�z�;��%��u=�� � �6�4	�H��U�ÀH��*����P�J��ӳ~l�,Ky�~Nd�q\S�z脈D思`�e�'�l��0l����A�!aur�u���ȚJ�ǳk��f@<�Ih���!^}N$���@��;	���4�g!���|�������}��5�Ǝ���n?%=�,�_-��9�:�x�V2Й��˘��!����\�R|���B� q��NAZCB.!S(c��^a?f�IMo;���Vr�~�ǿlE6�!�L�d�HV��m*q��iR��xǜ���D�ۥ�<L �>X�<r��B�G��fZmc�B�Q)q�a}n66��f�^��L�2eP�����Ԕ�i�$��a5Z(��.�n�������I�gٜ���w�cS�%�UpB���P�S�BlA堳r�.��}-��i%�ǥ��JV2����Nz�������,��њS���9 �h��۷)$At��B�/�?�0�R��|���.�r}]�Q���a(�N~RBZDQ��ˑ
k�}�C<+S�?��r���(�܊��!�����;�ۂ���	�Je"]���J�8Qh&��$�����D��d�y������?�����{'")!%RQd(�(ʘ!�2E�Ȭ���y�<�3�K�F$�)hT��{�}���{����y���W���a�=���w�M�$��R"V��N�ϕ�a`�c?���Ϧ?/9ƺ��T����SQ����B2��?=���P�������Ϟ�x����5�tD��Ü#y�H c���ob��S%T,���QO(*[���K�W���-&c��=j��׷?P��B���^+n��;�'�m���	�'��N��1�攈@;~�Aevdc��?@`o�H7j|y�s�&�/�_	��3a���ºW�N+�&�&�4���]�����Ք����檰){9��A&G}�Aj�bw�^ �~���k�m9�$[���K�ݍ(��X��`�;>�I͡�)a�r�(��КZB�ۜ/���.@���0��_�����~Y����`>I@o�-49n̷v��v��cA�]��������������%F�f��y�[፰��=`��Y��i��i��v^�u�Қ��Q�ص�J������|�*/��@��ń����g��8�u��r!�n5�	��%�1���/����©]+BC�K�Ԅ]#"��}��A� `��(��A,�� �=�%�!'&����ݑ��6�8��y�e�]�6gŵ��\�m�Q�fk�~q�|�s�-+=���~�M��Y;��<=�>�����`q��7���6�3g�����ߊ�٣�Uzd��|Q�z\Ӳ�����V�����Z����6�����E�C����u׃���=a�Y����$��Ma햺}��(ū]�S�:!��K�j�(�|�{ٶ@ݼ�fp-0t?ï��IJ�wx�J�糫�%;/�qU9��E�¹k"Cb]���`	���ڃ}x���o��L�@o �}��L�'�*?A�;P#�9z( �'H�<�p����Ȣ����0���@U/���B�dS:��W��ڍ��n$�P��Ur|{���<�P<���|�a�9'V��Ы�������&�����P��g:��O�q#�=��+��F(z\����Y=52�b�����=�du��{
��-P�\~f�9/�N��OӡC�O^�YT�g>P{G4�RS3�oƈ���qf��;�������m
0�.�9�����wY��>ҒU�ک�u:�G�=W���d3�%_����^��7�����_{�qI�I�v(�Q��0��Y���S��U��0F��L��X�|��H���?�#S���\���5x���W����n�dð̾������^ˮ-y:��������A�!/�������p;\]�>������Ú��v�iZ	�]�{.����E���涱��L�`½�M���**i����q�J���i=V{T��v��W;�s���(�B�'�`@y�a��(�L�#���1��'
�l2;຋�l�-<
�vPl4�cU��"�*�*��L���?\H^�i���?���u�O�Z�T��|H����M�P�&��3-�_5%���&@y:\����/�3�0�L�
)[�bel�	�לּ�&����W2EQR��z�q���qM�q�|��Y�\RR��[Ƀ߫1{����9HbHb��U�s�o7yZ��U�OV�p6G���Tx�yY��u^b}�.Y3V��Z G:��IO(P�ҭI&D��:M[��F����U�"���y���d��}6Oʧ�3�ٲ�E�S��m��׵�N+��7A~�ޯ��ŷD��ݪƹe��F�A��fR�_�\�}���̦G��d��'�����ڎ�~�E��=wT���d1�zY��Ez�{.�7�ﰽs�`�l�����s�o��l��{l>[�����,K�{|�⥛���[ّ�B�e��ۣg�^\)Z(h�������f�כ��"�O�hQQݫ]�i=O?�V��&-�}��fG�	'�mݡ���)�Tv��=�%��3G�X��:N�MyË�����U?����v��}�4��ǘ�n�M��\$�HT����:=��]���x���K5���Zc��W�����:kN�Z���oI�H�>�«����M�]�s�/��d����l��w�w&����`�����<��:�[ON4�u���b&�!.��w���w�^ξ;[��I��u�8{��(O:�~ߞ�X�����6�_�F�d�E�G��F7��On��wdd�2a��w|�����C�9���7�g�i�\l�����&cv�_H�vt8��;�x ������bV��X�ѡQ�:�j��$��>�\^��4W0� Y�������9��M�a3j(���.G��O��H���o�(C�����#G�X�׸��J\qN�E�z�_�~�8
��혶����ݶ�_��p�������i�I�ަ����kk�A����XZ|��"Q!�cT <,���_�U�Rۉ/���a��2Cŋo����(-�Q̶M>�CєU/w��p����d��w]�<Y��l���;�>��2n�xf��6_��G�#V�Nr���I&]e8���C�Q��{��~�+���o���Z�z�K�< s�|���c�+��T�%��U���	{���.���*^Z[�˛�R�My�+�Y,��rYJ��~s�Gb�L�SSa^�#�/[��꼳���qY�ҕ����rm0<`�i_Z>+�f���.Y�-��cg+���Q�u͑�������7suϜ)l�B�T�GJ��z��鹿���(�g��#f°����mL9��c͛-�f�6���`�k�G�v7�>=Go������aV_
d �n��-��&_��钆 ���;A�n3gJr�,��8�H!�`a�9nN�W���x�g|[��V�Ȝz��$����bhK���s�S��{���O�Y��RX�kEt������������~۠;_�Y'�Dw��ǅVwD��p�v��bn�372�$t�������5k���ٟ�"~C�%���m%����&;�Ơ9r�뻻c^j��9˱3�l�}�n�9ٵ��~A��SӨ�����s^A�陙�S��X�Åy�q�=T4˚*dG=7tW�����qQ�K����R��|���Ɔ�W-XTg��BP&9���5
փ0QZ,�i4�a*��ؘ�9R�HL��9о�tz��
ds�z`R� 2ˎ]��Q�0�v���և��]hA��l�Ձ�À0	� H�/$p`%�g��q���XQ2Q����hA�>(m$d��p
���@��,�
N$�s���gA���J�/�(6�I%p���Yi��.��h��X�h�6��;�����ݣ�0ڡƬ��� �s����K�J蘰b0�-T �|����?X��g��'��V;a��7A>��<�=�yT���2�(��'Fٳ�(d���\�;ڰ� ?J�aY�߇�	���-+�;��,�Nk�ȝ�*������ ^^�z��D�;#�K/�J���q��=��� 3��0|ؔڿ

�z{1��Xx� �;aԅI�v@���EPc����m��2�K��4��0�.�y�K�BAyH(�� )��N���M��E��2ؙ�E}��Р'�� ���� �7��0-�����E ���0J҄�yp,��ѱ�'�m�� ��D����߂2�I���T�(�*뎿�X�xe��τX��Yп���P��+���@��3�I�i 	�St@�D��8�i΂V �;�k&�O�K�P��x��1�
��G���h�,�@s���DP6>3.ي�`�N'�� ���bW�AH����06����:,�O�?�j���Q�z�����%`�
��@s��Y�y�
J�NNk�zR<9��o;�j��uj
-����%"i���t�����7���nM�5�gy�3sߢ/g���q�-8�#Y{�[N}��ۡ��QE!��s{9���1�v���N:���G�]�E���~�Lub�t��}gNIk�9�sJ���g��/2�_�wq�����]{���=⻛,ܺ����֌}��VK&/�[��� ���yΨ���aό��9ϯ����뀛c����ޟ��_~�F[d�(n�%��f��͒
���y׿���:��9gs�7lfD{7�B&������Te�|-��,T!�B�b�5xG�d0@}]�xûp���.�Y��*_���Φn֫"��u;�������]L����F�0�o'��H�A�R����u���2X�tu@[^��r.J�rl�����+9F�U	&� 	�K�#aA���:�R�"��!Ux,
�nD@�f'��?�:R��
W�4�mhK�J�"�IJ�}(!�z������c�4N@MK�b��H�7\��\�Ц�@~����UD_YP��zb�a4��iP�i���7�=	��6�
g�. ~��NF�o���ׅ1jwn�����79��LgK��Ow\A�����<uP�5`|'�	P!�V�>qEp0c
�˳���&;@�ֽ��:�B���8���@Ͻ��Q���aG�tv�{���<��^4b�K@;�j�t�z�1��ĜS��6w`܄��$���b������π&׉��7�`�Y��l����B9k�s���v3�~5\t��!͂B�͐y@�~�"�b��1f�*���rڿ�%�̃���9��� ��{�W13��E���_Qd��4HӤ/� �ñg 8/ڲKWYɬK����N#��m��W�v���XL$�G��%�0(� �y���=D)z���uG�w|�r����6���HI�zk<�
��!I[���q��&��򽥌�(z�@��$s��MJ��k��4��"�̄e?���P�9��"#kΝ�M��wRv�;�}mIң?����[(���>T�u
�S�'½�s���|#}����{`�]���Q�]Ϛ>�V�(�f��#z�Z$����ް���Ms��Z\�@���^(%.����М����z�`�1�x�g�|�ا��_�R���~`�4���8{���}�m��dU/ܨQ2�5�\+�@3cFQ��	���?TW�j@�8$�R~U��8��QF|�$��=���2�y�� �6w��K����8�l�eћ�8���'���F ^O��P2 N�����a� n.G-�h�����%��O]�Ns~m���l�Z�1��X\nE_�[����x��"�k=�����5^B|�~�I�8���Z�G�$H����4*�^�o����7�!���$�f���l1�6���;A���1\��}fLy�Aj��3�f�Y��~�лtQ���|��׬		��k���A�9����Oա�[jhWGo����t�A	�&<����n%k9\.ݧ�����R����1���줳����i��C1������D��)�x���ash�NPh��^���q�K��E���i�Be�`�v�X��Qf끛?�Ǆ�v|~� M�C����g���-� 0�g>��q���F:?6K�<����&�ы��,��"�I_��_�w�&
��t�`����( ��W����݁4*cٗ�5� ��xb/�E��ƅڠ�����B�fI]�w�AJ�|s=Z�|u�f^x�t�u���@�6�uv�"�Ct�|���x},c�qq��;m��i����l�|0���wm~���o���V��Z�6�$��ԙ��\�����MF�xJNXwJ�u�iX��sܲH�㡏�#���,����]>��0;i�c���۲��N�:}�ߗ�����# �́c�5V��͹{.3�\���)\�%���N� !��L�~�,v�Yb��Q�Z��A����*T]�4?�xÅb��kN'<ϒ�9>��_j�Ƅ�@��05�K�
�%�w�H,�	���d,z^�F�y�^�H���� ��s�NQ�:n��?�����Q^{b��_���A��2M���,��!G/�1^�;��]��_��"*H�a kI�����&�޽��ׅ��F�����G��l�V�s�b'��>g�ʻ`���]��:��댁��7�oֲ�O!7���i�C��������"F{,zi�N,����v��w��0"�=�~sOAF��<�1����o��i������v]�9�8 �^҆��M;vl=��o��1�	 \Hz�IP;�i
+������	z��5L�,�I��\�-���F�W��L`�P*���tp� ��
���m��[1�xB!B+���y7�Ǉ�d9�z��ǿp�8\3NY9NXSN\^N\h�K���3S��ڙ���.�f=;�x���%16���V =��	Ci
�����1ƴ�s�xy>N����r�pC��̗ۗ�.o���w*K��%�Ǯ��t9���j���*��k�$��^^�mx���Y�:7
���jѣ>%�q闩ߔ���x��&1*~�lH}�&L�U�p��cM�Vn���_B�_W���P����@��7X�E�n���a���9�N��a%^���}�g_�[�<������$ܸ�Α�����{���褚�_T8��tm�
�q�_�5�/x���	_p�À������>���`F9�+Cr8��_0��T�B�*��#*� ���:*�'�7`Aȭ<?>�}���&��q�L+� .k�=X}��w�pg��/�m���@d\�8 ���q`ũ�wŷTG��\�{���Q�GWpJ�ʖO�wi2�n޵V���J��ʉR�\��<2��
.��)��#g�R8v|9�Κ�0�L�_Cz��Ô����z���M�i0I��t�����I�$(���D�&@9���I�[�������0���ӄ`���\`8�� �;i�i�vFpP�؝�Z�:`����k�Lz�U� �(�BO�k��V�G�ҿ�
\��y�o¨�9�>����ֳ�����m��B�?��>w��}hz\���Gh�����2��\k�/9K�3����ͯ*���*�,.�}�)~��K�@�;����:�*>~L�P�=�j0cÎ��of�y���T9E;��c��=�u�e�򸮩v��M!Ip�[�YZʖo,�Q	[uT�t?�cX�o��Pa��m��e�9������������2}o�����\���ֽW�AnC,���퇦�趖�����e�@:z:S8�����W�.���?�~��_�93�s��{����>%v_D��r������~��V�n��h�A���[h�(��E�>"��,��
'ۓL� m��I�����s�F��Ų� �7��<���,X�����}Y$�o=0���I�����t�&����+h(G��h ӝN0���� �q^l|	��wN��^�F��b]@���DR��I���CHuX~���JuX\��+
��Q{A1v�������$���8�x�9 ���:�v�\.����qJH�v9<��ZB��<!� ��ٟ�>��U�� �! �>�0E�� ���x|b)��r�����9�M�{�^w"��v���-��� �W��z���6bP�~	΃Y����tF@E�g��Msl8�A��@������T=?�7��_�������0�H].X1��v�[D �}#�5,���X���!"�*_�bk©�W��Q�Ϸ`vY��>�D�R�a(��'����� ���1��(�bb�$���I
�?�����W@�6��_�y�{�S���|���n�� �Ϣ�x����%� ŮFo�H-���V�F=�"���UhF�4^_��N�iq���Z�[\R1@M��+��a��a�ν	�`������}A}�+faR�!z��RC�d7��&l�_:�#�h�)�]3`t���*�K}7���>I��H �z$2�ey+�l0�H$Z *�h]'���"e��J�A�|Ք,�$
�R���,HP�>��������~���}���	�en�u�(R¨�&�wi��E�fzٻ2���%!Y��-+)��eȢ�7p��j���������Ղ�(�7�6G��l�l�G�F�c��F� .�)3��ܰ�N�P_"y�:Fo���ٽm��K7�!�hYME�-���4������dra&�����S��A�ŐZW0��e����V߅���6�����j�|�џ�X=U4�����[L����+���[3?P��??4%g$A�),A� K�4��|D{�b]�b	�唆P`��)C����L߁�W�����Rt�/��s��O���M����Ւ�9����Qiq���I1�S)�D_��\�;�c�FRA� �[��%�,Ҵ'{���p��LI�	��k��^���������ԟ�r^� ?�&VR�)�F����	
5`�_	�I���V�AQT����Q��m"� �W݈z4�|yjϭ����wY�gZ?�y2�Sl�Iv�Z'�-���*��uJQ��V�~�d��6d���	,3����]�����`����#��l[��g���oGzR��7ȓ��W������d	���/�xo�Ś%��?<uW��J������o\�����W�[B�ͼ�X򶭺0�r��Z���n/r�r�w�Ҥ���g�#d{��xҹE|�U$.�ߙs}el��ay�U��u{ʏ����I�G�w������V��uE~����� ,@��x�WD�tBA���~�{B�A���NJ�:A>�ʹo�����U��d�:�T����p�6�$�n�W�_*'��x�{�����q�!�W`�v�:h�*oС k,�Ha���m�d��v0���|'�bD�bG�<r���~���٦mD���n�Q�y��̩7�"zĐ(t�*XǶ�O�{�	��1���`�� �X�E� OUk.��X���1ip$M
�ff�ud�-�D�q	�y��C�Ib}Ȫ1�لHM��r�+Y�*��r�w��:�.3�,6���U*�0@>eUR���J�Ţ�ם��(4�qV���ܦ��Ƥ�{������贻���]�[w뾧s�nxt."{ʻ���Q�A-�^i;�K�{Ż$}=���_ڶO<�\x�g���&����]zg��7�J�%.��=�+{��
��wg-�:�R��خ��2�Bd��,}���~-U3��7�"{ka���VL��Ì���^Lg?zu���!L_��6��<s����QЫ�&�%L�D���J���O�} �/i ��
���7%��]j�6dS�-�>�������D�yf�UD��V�?T1�p����$�@�)r�Բɹ059�P��UN'��PB�QLp;p���	 Fq_e~r��s� r�, I;�3��@f찔.�"ө6r��� <$�"ʨ,=(3���E��n[�8��	���&���^���=P�=�����I/����#���e^�P�	�Tâ����/���O$
��=,*�0j��ؽ�� |��W(�^��Y��~�u�~a�������v��Ͽ�L�زt ދ��n�d���Z�_�0
1��21>�e�O�A`�P�>���3��C�'��o�ȑwG���GП���"
	P����O�M�Q��c����d��r�2�$�u
%@���0n�ɉ^�N���9&��B�1P��w�߁A`�sm��Nx��")�oq�0/�L�݄OL���X�d��	L &�@�H��a�D~:�u�O5���cyK��89��L�<}��@�0u���Ԏ
�0����t�$�P:-8�	�q�z�[��$}^~��x��ڲi�@���Q�4.8����.��Sr���A�J�VjA@�z	�ObrXJYHQO\�@�(�11R�=�[vj��B��Ye�w)�P%��� �����{���`��I�8Ӄ����a�0l�l �-��k5����u�LH�Yᇇ.>��@��`c���6y��_����m�Ѽ�Q������F��|X=�9����Y�E�+_u��)����l�9�W�&�[S��,>��U��s��3�:�����/��v�u��Y_]#=��Q��z����wyߞ���w�K{����
/��{�,�,�0Y�$h�s���-A�xA�7�w�����	"$�.>��V[����쁮�:I�5�F$�`'>*���	nX�����{�.�979Ϸf�rX�8����G�8�8��^t79�#:3�v ���N��X���&s�\�#��5��
O��[!O������@��0lB+�KȨ��Y��&����)�4�(�8�����0zZ�UBŏ��u�͠.�㉥H/s���K�s��	��+�j�ؼ�E�Ǳc�����a0bB�a��;l���2n(~�k�NC�M��2~Ilt�	�����5B6�G�i�lz�#��~�j_���YbJ���������X	��<wةn@��w��z���b��Fo!������S��RY{�ݻ�0�?%vc��.��������o
��\�H��0!�����x�y=dyP�"��T�o	峳���5����`��s�+���q3�c"f+�S�
������e�{O�*ݦ5H>���`_�Q��z���c���zUL�Z
5�Rz�2���u�(��;`�g���0}q��@��ń����g��8�u��r!�.��r�ɘXI̓AS
��}�Ԯ��C�!3EGp�u�����09H Lp�z<�%����!>A]�~ڤr��\��<�ޞ�z�,�ç��LB;.?�kB�Gfl�{E�u���x��Vew����'|��ظ�t{ډ*͡�����w	<��<�/���v�-I��Y���Gm��>�5�F�꺶�I���}g���x��m�GZ��n�*�r��^eɳ/g���n��pvnۍ���b�"B[e������p|��Ưv�9�_��{��Yz���h��H�)���ϔ�����8 ��C��� M�[ڵ�������_U{�ܫ��IQ�g��Oe'U���/؍}x���o��L�@o �}�ذ���S�� ��SR��!D�$jM���m�t}�����ә�tP��FP�_��l:������1v��	}n,�Ty�|+��s`vOύ��k�b��Z�J'^qa��Q��B4����ܸU�ѳv�՞�㉂¦�����e�N�d#ؼk�<T�8���Q�� {�
��(�#��{�\wC���n���<N��ȴq0�ض��g����b��0�+�ؑ��}{`Q���sM��7��;<~��N���N�X-�|�`�#�<�+Q��G�*>5�ۄM��X�qd޽K�m{9��L�_<�v|sË��8|�縹?0Zi)�W��&��I����'�67ڼ�ȑ(��5;pHm4��zF��W��1�ặlޥ/b�J&,��h�:�<C��]�_�6����v��Q8�V�qeCȵ��rߍǮ���-��;�m�ӧ a^j�e=sh˫�I<�����s���~��T���������̤��ܽ5�N\|9^��#Q;���v� Zg+h�~أ����/�0���C�>��i�n_:��pl!�-v_�a>�U ���wai�� ��P��W���L�wuU��%�w)�I��z��j�w\�T[2�H���,`x���k�a/�c"�����L�kr1:02Rm#�c�V�&0���w�sLwv=[�~mdd��w��;���3a���`��ldSg������Rb>/��	Tx���ByxnWZ�y�C�7d���i��<�����"��~[^jD��bk��j��-)q�3_v�b���FFا�|x|ʝ=�ߧ���C%�|xo�L���c�z7R{����CK�7�	�	�
�4�./��2-��Kvה^�pM����oe�Z��P��~6�v�O�.�Y����|o�=�yTVfC���j�@���	w좺$c	�O!����9�,+:CȦ�Y'{l�p;sB�"�E��n���[
?��gj+�*�[�ï`A�Mz
��ޘ[�I4`2@Re|��D2���}\��Ku~K��K�9�����jW�=�c�	�lz�����=앢�����1+�����z�f��ؓ*�f6���q*��Ŵ\�2p�'����~�$zS\i�c׈q���ξ4j�ݷǜtV�K�h���9EN ��ٯGg0��ga�+rv�+CU��7Y7�=q���?��1��x�/�)S�.��d��k���j��ǭ>�e��Vx�Ze���a�ח%�WKO�yƐV��s'wQIyk�ޡ�ʃ2�B[�J�g���Q�ٕ}���4�"��jm�Y�]x�p�������*}�1��pR6n�J��>
���^�|�q��,���V�2�bE� D���5�Q��l���7�,<�*Ht����<@(*:������B�@q��˭��ۨ�\�a�q�$�����'����-�N}:Q�&��1A���b��W
!���g_0���V뒻{���ܞ���y����}��3���<P"�c��Y���Q=֘7�G�T��g>Lw�0qS9�&�Ev^����Ku��.�yB�ty]+y:'��,�BA��$��`QᨼN�	��p�CzЙh��� �����*J ��J���R���t|Y�cF�cX��#�X�vZ�Rg�N¨�I<bB���^�V�TM8lwH|����:����~6�-u��mGj9ۯ��G�eٸY*��B�A�T���%���,~V��m�"?�/O:�-�闾��g��.�}@S`��y�ޒg�6�s�88;��0�����y�M��mNw�J|�>=C��֧�a�I����_ڑԫ����j9���ͽ;V�mA���گ�Xl���z){��N�Nخ�T��T��}�{/	�+�I_Z�a�DmsX����Y��b�[9g%Ƌ�>��)����np]w�N���ê���𫀸&��wɽw����K7ߴ
�~����
�r��;f�`s���;���2���[Q���[���H����'F�=�����u�w�5UR��s*��A��4YJ�h���	]�T<��V�j�O��bF�����.	x�{��e���3W�ѥu���>Y�K�sߨw���-�Zb�����i'�ַ�e��9������HA׭�3�&�p���D߿/�q{I�Qώ+���q�x�x �;D}���n�M�|N�ԟ1Y�*.��q�g�5SI���1��̥F�S�5U��Q����߼$ �ag�6��	{�f)=��1����M�����*ŗ��q��֣��I�*��"yv���9������>RzܨtE���V�7<��]�/� Cy��5�"�Tٳ�6�x��"�^�������3� ��J�\C��qYټ��j��@�ZY�i��K��.kz�x�M#�p>@����,�s"<)��V��esL^L��u�%�Xb�,A�P�8pXg(Z�%E�QjI���쓁�FB6�
硐z�䀈ɂ!0����B	��$+9΂\Q��*Q,t�3!>����d�$�&C%E������D����Qj�h�rx�H~��F����cB0���K�5i�Q˞=�A>��M��dC7��|>!��<�JO��۞��\� ߓ��'43�`����%r!�?���O����aT/^���A��aXS��O���e���M ~'�������ӧ8ly|F����>�b�Ŭ2	i���0���O2Q�Oق�W�iJ2���O j����)ܪ����
�t����a\,�p�!��_�k�+��w� v����R�.&�L vF���S�4hйij  �v��~�y�#�+ _������ܨi)��JFb���9���o�:���{��ђ���'0���ࠃ/�����!5�:��a�y)��#XMc�ot��gF��r 	�St@�D�C��������,h ��fr!��d	��������.j}�	�&�B4��Oe�3�\��p:a� W�^��p���#�O���z��j�U*�-&�j�yFPs���aR�#v���� 4��o0�����$��� A��'����s��op���R���Y!����͑�--�ۘrL��WtO�K�gݶh/��3ZLN�#85�.���&��D}��I�M��.�B�5��r7X\�������7?,3��9�mkvb�~���߉Ռ���Q~c`J��M������7R��W	��kf�Z��F�t�������>��vG-�	��~�i�F^'燔<rӬ@�W�]O^&��>|��c����Ξ��T̤�q%���X��g2߸���4]mK$CJ�J��ǰ1nE?z�0�xw�*�W�l =��B��Ţ�$�$#�'1�Q�Y�W���ĕ���w]GGg_t�*�<i��nv���By@|F�6�}|����V0Q=[�o`\
~)�l�P��`�?I��y	�Y����?�'�K�*���=s�����o�JD�|����z�i�(�O�o'[��޴S��}�UY�(K����R͢�~l6�
E�?��՘0)�
JHH���*�I1ZQ>��[��Gl���کyR͆����K[W)|:{?����먤��3�Or8��ue��u���ƒ7��w�Le�٘?�Ł���;�"1�ӰC�y��ÇߨX�,�ҵ��,y�@��N��o�;~ޙ��=�n�s����wNqU췻�[8�H&��l|���?�#�'��o[������U�l��z��7^�q$A�ߛ� �{�i�
^������3�@yqwa��`�#h/|�*#0���>R���Q����>�X�6̻���W�v�b�q����qcN5	�W�:(s�f��إ�&{	ؾ�fF]�`t��-�V{��@-���A�4Y��K{`�R!E@��1��=�+ܾ��^y?|�yX��8�\�	��.��T��Pm�K1�m�*,|��(v}�r?y,4Az,"�~��/P��P �g�wZT��3��H��5c����>@��p � �f���uT��w=tM��*#y1�N+�Be�aH�=u��_���̜�22˟�]� ^Bc�22ᡠH wֲl�`*O��^�����U����B�WγZ8e�h����%g�t���i�➦� }tR�p=1�U�؍��O���jQ���A��ﱃ�ꄮ&9��0
��#h�}p���ƍ|��cE
v �⡞]0܀Tf y�2�7�y(��u($R��WJ�;��#�C"႐�Y@��{�,ܯ����.�Hi�8���y�:@����W*�Ԃŝ�QA@-.����|���V��N@��]��QA���BL@�A�(E��9����&�Xص���hf൤pxi�Z�j��zH�C���K���3ڳ�z�X-{���e�
�C7��g�n�~��l�]��Sy�C�k�MR͡�j�ul���E�R���VI{?���e�z��s��-���rk�5����gY�5N���Z'���/�:��]�������.l=��֮?�y��#*�!�/�]�*�DՇ(��z�s����;�U�^Y�9����C����RK+�G�?g�J�[<�Ώ]��6T�-z�dY����A%�\|ca�Mg=u��Ծ�ݸ��cئl.`[7Hi�CP���k"(a9��'n^O-�(VT�dp�x�\�WE^��8k���`�c�X�����h西�f���Z�iq2�s���c1$`��j|���Z�G�.�!�jJ�FX��?��8��:YӲՒu����Ec�[-��b��$�����t��s���ї^	#���'��Ɥu;V\��Y���	�&�Z!��A��?}��v%��E��,������f�:J�f�ֈ}�6T�Y!)걏�jpAn��à:��q�}� �$l�(y�b`9�.� �eV�|�n������imz_�fZ`�.*�e��w)�h!��C}��������˱/yW�V�uŎ�3��
~ѭ����3\ſ�O ���3� T�8��e�����q���q��?����M'�K�OF*jӛ'�i��Bh�GzLH�F��G7<ɋ���Oฃ!_�-fdo=vĖ�`hG�c�qD�V��߀�ft�r
�pփ��z�/�=�fcx`i6�p��H�Qi{�fp;���E�j�~�ɜ� �F맥e���C{���|��ie[�zǾf�����	OS�)�Hg���1�Y|��zvN/[c~��K��r�m8���� RV#ʓ&|������k�q�}�z-�����ѬU7FK#�ӄ$JJU�go�x��sJ/��i+2�����O��$W���^sR�>���kGt��}���Ƿ��?�/	���mz�D��5B�����,|#4%�[yw��/�o��j�Ƅ�@������;�@�+�f �8j�^ �d2��<�� +`ɳ)uL
�F�y�^�H���g/�&l=��� |*�X�/ய�/�s	��ud����ܚ�8�{9˗f_�5Z�h�~'��T��� ���e=����i��f�7�kN��a��>��#���0������/������7� �B�Vni ��k,0��A)��0���c�s8�aͅv,�`m�w��@o���]M%���իaF�y|�y~3b���N�/Q a,ȵ
f�_�_�7�6�Kdy(��t����~������҃�bpNn�Ã ��qL;�`a��# *
���m����4��m���Tw�8�_���52�()��b
�����o�6L(�P�����z$��>>�'��Ԕe��^��\3NY9NXSN\^N\h�K���[���UZ)+С�zv�1�� �W#�)hE� �m�HJS����<�B62t�	.�y������������?k��;�s�ә;�D�5�5,0P��$k�U�F��'��mj��~����d�+�Mz��ʦ���7�w��p�v�dR.)��U8�Y�<���v_2�mm+<X���lC�}����i���N��ě:/W���O:�$e��U6�.��%S�x}���h<�i�2Ǣ��M�o$����C�޺.�i�:�O���pl����%鈴�������Ǧ�L�`�ka��h���F�D8R���9*V���=������� YKt}
���	�?'[���.|���� ,凜ɵA<A����"
�L^Z>0M�����o�+Q��ѿTZ@�� ��+b�9\ݍ"K3Q�o�Q�kn� �{�E����0m+מt;O��p�h`�����f��ylW�`�(��?f�QD�w�.�
:џ�]�¿��n�$z����\WD �m0I��t�d#He�I�$��_�?���r5v@C��:K`
����8�^�x�ü7U���sG�E�b� ^!�M�%��_:�\�!��}���;�Z%��^�a
h��s���Tn��1��u�����C�v��`!��� p:��#��9G�v���z�s@�Rp��bW����t��ZM��:�#|�2�lY�|���휙`.t���̖8�;�y�7v�vqt�p	��;������|㩥�ʂ�{8*"����9mOm6���q逜d�H�#7}�^��߬j��e�2w��A��k~!��E\l��f��^#��� �\d`���k���"�z��Gm���r= ��p������خ��b.h臼(=�Ťŵ�����A۾Y���Ls��ɕ�1g�`�s�����]t��Ze>	zX~�8����q�>���r���f��΍ E[X� �S6з�p��Yg�4G�La��Z@�Y@�g�t+`~y6pj���̱TH+���L'�k.s��JPߔ�6���UGea9�����v�y@�R���Ǘ�w/@��	�rPN�L}�@�nG����~�B�$� �E매 ȧ���[M�__�U0�09���f�r��7~nE���|>X+7S�V�����U�,�w�9v\�}��徐�_C����)��!Y��j,��3|�(!������9��a���)�nm�ת�Ա}le%��;���&�{�e���lyװ�$�ߌo��ϟK�,�ٷ:��M:�Dr��>����X�n��"��G_��Y�h�X()�������B�R���j����7�F/��;����Ǚx*��ᙎ�6W�ա?�M�=3r�+�n���p���K� �]�.�ȝ�tB����!%��W�! N��P��a�"K�.�p���N�`x��Nx�!���t�RH]�/AZI�(m��J�a|
i^:�[���d�H���b�7�����b$��V�=$��	��b䳺���u@ ����H���+�R|���R��F-b%5�C����������^�@	�� �N�_�R�
��.����>p�YQ�3�{	�Y�J��g���2�Y>:`Z��,`���
��/@5d���͇�u?l��	�x��k�<1�H��r��s�]ގ��< �?^	�Ӑ�:S�AC>���.u0����O�N+æ=0��,ǁh���!L.�0���`nz�d�`2�w��9�DK'u��OR�x������U������(��Si�A�Vs��Ŋ���:�)�&Jtߊ���Y<�'Er3�bQ9a �M,���dթ��;�u�H-ˇ7hZ�*�5�ѽ2�����k�� ��R��|��c�>atZ�������$�# UL0ׯ����a�uȞ��84�AI9I(%��#!�N!�M��Г`��~9�Q�SQ@�/=���ᰖ�>0:@�(�����Yfk���7�f��Ѩ|��/kG�V���/؞�G���8Q�t��J:y����Y� �:=eFO�e����th���?�#�A`�]z
p`X��J���[;��tH����`����i��i��
�!���g���ހ�d%����"���p��k�6���~	Z�����oF-w��y⅄=�b������!�m!|�w-��Hs	��sqŦPa��V� ��8D��/ޏ��"'�����@W������I�m�)hxz�>���<iAo11�1�m �	�(S�8�c(CSrFBT�"��
�$AK��!C��!�E-�PQNi�%y��g��T��(*��/Σ>�2+�]�����8_ח� ���CeFRIvXXX�Iz���)LϡeW�\(+�+��/-*�@[Q�(KN���483Y����ť�q�gJJ/ ����T�����y�>Pr�yA��XI�$��;���1�<`�_��;c�dAQT����Q��m"�N���WT�$��EFF�ĝ:������y���07';=96�/$�7$26:���L克��3e��9��s�����񋌊����JM�C������3����I)��I�Y�9E�9��%e�e�)�	��a���SAA!���^	��>���씵ts>�������h�r��i���<b�l��������_����]\^z:?=6=� ##��tvFJltlLl\\TLt��bM
��9t<b�Si\ '�)���,���^����
��`H��'BP�|P8J��>@�[��*��Bz,��g� ~O�q������{Tț ����P������x�{vl�0L�9��m��B�܈h� ����D���Y��(�DMl��0r������Í�Ι��q@8��p�l|o��V6f��n9�ε`@�|f����/Cu7��נ?�Q�b�Ry���B�}�P���fZ�΂�-�S� �ݤј�6��
E�'\rZJrR|ԩ o^n�������GD���FGF��z{�:��88��ON�@�I�)I	1���lvY�:����	�HHNNI��MJM���HOKM�����M@���=R����""c�R�S"\��9{������"����k�C_o��6uS����Ĉ /?�㎶{�v�.�=OVM~9'���������������q�/0(4�����ƛ��;��"�ɷS�I���%�W����d������38�#K�������h5�i���ȋN�DId��(�  �$�7t <��F��ۍ�HX� )� )��ݏ���_s1q;�{����&.fc�*_fVeeW� ŉ-�]/��^�̮.����J}�+{I�sk��T��J�(�{5��/�(�Q(�D��N����D���A��4�4��d���9�1��d�[�r4 ��-���S�Yj ��OQ����}TلRh-�D�n�B�J�������Pa���y<tx��kv���ǋyx��ө�?؃���*�\�6��'��HIBsͯ�|"5�_���?&��qnǇ�{U���T�j�#i�<��E�-/����?B2���Qbi>�/�~�� �� �[����:z��H>q8�^�:q>�.�o�©�Dr_�Y���B��J�+4�N�o��o�+�rdd�L�{��������@�!�9�{z���oZXXL1�r
T=T��k��O>��
�T���v�$�#q޷O������[���zaPH����3M��$�����^%
�R�e(�G*L���ž)��Kt��4jQ��2��}����/�p�>�[�d��۾}��/,�o�,��u�Ľ�%zC�Zw�g�|�6���Kk�H�V�|�#G~A*,ulu�vLB���v�QD�j��lԄ[R	G�+-{�4�
۔@2���%USU�|��()��c?� x*I�X�F�)��c5�a#��Vׄ�[j��D��W�(�I�,(m3��pV��T���E�Z5������|���-ݏ:	��#�ihh\����\�(|B��{���v�+������p�ꕫ�K��MFB��A��?�틎N���N�..��OOO�G\f{{]�-<r�����w��ꛯ�ܾy���K��>gQj�g�?22:777}na���������n����+���F"3��,_Z�J������f�'�^�0�W���G���Ν�yZ�ޞJ��}���c3��/\�������2+��ן؞|,�+<��,E\^Z�~���˓/�ܾs���k�ύMNDǧ&F���=�h���a�Ϫ_J�SWg�K]A��֪3T׵�s0:s�����`����P�(e��V�נ��j��W�h��tvv8��sC��d��Ki	0�u����m
��^_�Ͼ��27Uȇ���z�%_g37�������ju�˳�
2�����,�?J��j�{o��Ŝ^±B�p��#�v�� ����V�rK�+%�3Bjf!�F���?��Λq��}����Ȱ��p�u�=�A(R�C�0:!�y:z�z�̵�G�K$EV����9b�݃��%�;�/jF2�z��H��ZD.����-I�p��R]��ͺ��!����7H���T�rr�r���x1�K���Q_}��g �������]����c _)�~��b�{�(�?~���l-���Gm&�c�ۭ���h��H�龃P��կ �X]�ɷsp����g�/9�ϻ���#y�V�S$��
�Yl��腀,4A�1@�t���a��4��\��vd
m����PEa�Ճ+M+��#ۤ��l{��U�4kB���p!��6�� �M��&��
G"���ё၀��)]��ֺS����������P��鴵�75��6�BãSS���3�SSS(k8�4�s�f����y{��#c#��������且�����p4*�EG�'Ƈ�R	#c��Ƃ�,]����r�����z}����������V�`$���tuu��*Js�מ~AL���^�`��/��
GG'���Hdx$����{�m.���ﱵ��۬�ٟ�)�kw��/��:�8iC&�� h���q��+��`�հUN���4I)I�)o�F��o�z�F��E�P�:ʤ�E��! 4�K�Ah\���P(������|.KQI��M���Vo�ߏm�V6������	B��Sry�Ʀ
�}�p��z�6�ӷ���[����k��svp�}A_o�+��B}�>R	ncW(�Vt\�����������>�͋<C~���:�oO����7�����Z�5�%:�����bu�}>����i36T��JKk�Z�ܾP�� B����u�,�E{�=��+i��:\��6�W�!x:�\�޾��@����s���.����qv��̜��햮�=>G鎂���J���ӷ4�Smyg��?ޑr&9)��Y���j75g&n<x2=�7������7����awx�.�K�|.S���Ѫ�����R��V�������2o𢡊oI���Ք�[��N�R<��̽�����M�;A�h�����9�w��*����Hn1�_����z��N��zG{l��Nu�?)�rK��G�<��L�֏½PP� ~����;jw�_n ���2��j����?A�@M�+���bu9�y�����H�9�:�T����NK䅅�=?33?3>���苌�&g�a�_3,��σivjraad*���Vkzk����F[/j��yp��D�gd����;?���33�R]�Ttx*���Q��(��H8�kw��NN��(�����pQIYYY������0'󸄴#�~�w��Y�Y�yť%Ey�ٙǒ?�{`��������PSS[S[[[S]m����(�9~Hܸ5a_�����������*]q�t����H͐�⢼�ӹ�%:���������b]��*o�����`r����ǎ�L�����������⑬�2]��c'N�#�ؾ���7���s?�o������

��
�P$}iVi)���WUd&M�����8z���G��~�/_߲7%#�����J87�Ze_0���7�]C������Թ�����ёp�����	�����3�f�#bnV2NNK���!O[՞Rs��50�F��&'&&F#�	����$�K�����Htd\b��������H�]���lk7[�l�].[miAZ��
���5���?�������Ն�䤓G�%������Fwx`8�HA'�ǆ{]��~g�wd|4���#!����]��3g�O��/E���U�2�n�������H3h��j�-���.����^�Z!��W�O8sn؄��>A0��P�z`=�.�b�I�p�T.��1u�·�R0���;8 ]�C���?��>�"�5��m�A��t�C���E�T��O�}��J��9�vBD�P �cT5\���]�>��z�==C����E���!t�W�����p�%$�]m0��G�����n��"|�=�t������ĦV�����/�#�O��UD�>�����Z|��`���A��d�p�sZ&F c�DF?m�+�
Cn\��hk��mh��o-��z�o����G���=�g���Q�r����=="]�����z����j7��5X�>w��wh0���M-6SmI���3|nnvva�����t4441�ڌ�g�˫L.��(u�C���љ��������Ņىh����ܾ���htt4<�D'gG����h8ؑs�رR����a�z}�6���JcC���i���f_�Ԁ����NSquS]~A��goz歄C%�@8:<222�:'�c&����ӳ���ᠻ����#]ν��z���Qut���xB��d�����K���/.�ϏN-̄���R�B�%���=W��58�����-���!����xa�Gpp�4ڴ����$}���l�7Cys��C��Ɯg�� z�ܜ ,��-bߋ����}���K��.~�ڥ����pX���^���������ŋs��M�Ez�Ng�����r��/������^�ti�ꅅ���%-������-�K;�ϯ_��8�x��;_ܾ231}��������K.\�|��k�W��'�_\ZX|�bs���|}3S#�GN�ݖ+l:������N�-͍E��C�wW��(�e!�l*�2z�/./_����r�������R�g'�G'FF��G"��]	����H���U�O�74��D���@02Ⱦ��A�d囩l���p@:���-������J	�V��A�>2���ta�A�w�ڈ��og�-.��A�����Ws��� <��=	��I���D�����I�"�D"�l�x���d�MDǤ��� "�bdݭE����� �΂��Z�(}T��~��.jEض�fUe�k� �k�J�N,.$��&�J����Kď��6�-��&��� �R�ճ�߶�E�?J>�ߑ�?7A����������-��?�t�E�oޅ�ل_�������o�~�H>���^n:�7�H�H>�.��݉���G��ɶ��ܿyu#.�� �#�~�+$�HI�{����4E����t��}C���q�v���{(��W�����b.�~X�ǋ����q;��L<o�l��/�f��'�\췽��H>u��x�<O�ض�Ox�W�;���g᧔;���aV.�j����ޖz���ٌЛ=�R�(�/��࿳��Ͻ��0ϸ�^\�lz�?!��g'Crsf�%6բ��3� N��Y�q��X'�$��X�5BC~�EePQ�!x�]VҀBD;�n�ʐ��A���2g�J2�i������N������;"���P���L��E�RF�;H��L��AqYJy<�E�)^5�\���b3�Yb�y����\.W%y����l��e�\�RQ��C(�u)��q ��*�s�N�U��E�R���x�?$�I��/^��ڍ�K�Ã��ށ�^o�/����[D�N�3�Q�;�s9�#�箬ܾM�[n\\\��|a"�ץ+����s��ӳ��~��⹹e4�rky|t�����������KW./_�z���险�������Z������MOEnkn��e��PnQ�xt||f~a2��v��y]�ڲ��	�ꪎ��wFg�.]��8��͕�W��>���W�OFG"�������P��w��Ѣ��,�xlR��;�٠��^2�e�o�a�����R�8�gk@��f�yǷ���'sAn9F��o�j2�߫g� ߆�\֓�:x�����]�XE���#E�n�(o/���R�����r���7���(��R��%�8���� �9��#5l/\��|y���[+זgG���#C���~�<̀-._Y^ZZX���X���8�����_J@���+׮,L9�r�����ŋ��.ݼu[��/���˯%|q����ɱ�嫟�ܸ��r�K���o^������ONLM����O����t���z������ť�7�ΎO����ހ���̾=��T���;}ey����뗯����/o]��r�˯����/n]�4?>5~vn~��|�݆���`�7� =��?�v*��>�G��J��6�TXr  �[5����� ��	�7}b�{��r���e�z<[�>>V����b��3z�����jZ���^ǥEh��/�������a��0�+����<���Vx����[�� 3]�tg�'A�[���s���J|^�k��x���H��w��N��c����旱���S#�W�YZ�"����n����x�j��[�L��7;q��g����������&�����<���ſArK.�l������	���O�~��u?ݛ��2�a������Yx�lB�Y�ڷ�����+Ӳ�SU�g�k*�+KU$Ƌ����U/�2y�Z�ol�546V륟�L���1M!���S����ByDE�3�fV`��³�K��$8��3CL�G�V�U:d�OH�!3���4�u@�ow�����M:j����L�E[%IA�Bx�]�� ��%�8��d<$�>�`�
�|��^�O��w�Ӭ���C%�� ������.<z�?���p;�%���tL���W��^ ���m�Z Nn�����A�����n��B|��w����s�<ڊ���a|j�Y#�%�6�����?���I��~t��i�%�=7��In(b;���H ��MK�a,|�uB���J#јR3�ctt�{�ݡ`��u9lM5�����jCeI��b���ڝ�nGgG��TW|*;'�PV��34G�&z_E_��w�t���d����buu;��@o?Zx�G"�!_gW[�Y���	�|�/��wٻڃ}��g����F������f66��凿��nO��8��A��l���k��kt����?<'�����Z���z�Z�{�������ޑ�~��a��[��f��?.6��冶�w= ���Tc;�1R� Z"c�#�4��Lܿ����p GE��@����F�!T*8K� ċ/� Xxvu�8��q�GV"b�W,Ã\�a2Jq~�+�+�|u�Q����6���fvu{�A_�����w��;���������
�7���8CL���ؙ�Z!t�m=�����77::\P�Cb��҉v|ōAvjEk�G<=�f;vuw;%��19��R]R�X�,�c���%�:ݫ���Fck]Yy}K�BR�,�u��j�N[#Ru����J��h�XZ��Zk�6��K��k $���`lUTzn�Ql�Z�����X��]Kځ���eص���,�3��4L�6[��M�3׀b�q\-n[��&�AWmIL��g�]�G>q�C~a.�=�ErNv	���Y$���P:*p�A�����ɺ���,*�O�w�x�?~{+����H�L�V;ũ�,��,���R{f� de����f�LC��l���w�9r,�X������3'O�&H�HܵcǮ���ee���eee%E��y�'�vo{����K:v<#+���c��

���J���J��33�S�Og�I���/(*.�=q�LV~aqa��ڋ/n�u�PRrJ�g�?Iؾ��MϿ�ܳ�����W��<37�T��䤤�l۴�������G�|�k���]�33Ϝ�8u&#;���(;-''7??����0�Lꁏ�Sӎ�8z0!a���w����_����S��y��,��\��������E�$x��CYwA��)�u&P��)5����\�J���C�А2!�ك�"p����Z���h����6�1��#����m�����`l��6en��PY.u�k,
� �kp�Y9]ouH�f{FU�]�q=��>��3�>x���n�/�a����{�-x\���O��A�OC!�_�8��a6��"�O�x��ƽ��@��H�"ο^��+_i4�{��z]pI����ڑ\�j�{�_����2�]0N�t�B�h+nd$4�By��B��M5�o�	�B�/���/�q�>!==��?��Ӌ����{�����;���{R?fߙA`�Ph�������,�hL�-�G�]�.���q�*ˬ����F����Ɉ�fMZ������u"����h0EyΌh��p��q��q�r��r�As����_�	������������3����p$G���HNA�Q�N��G���ncSCm����CuaqU��ln5Z�^������6��U�5uMu6)G�����W�@uy|=Ζ����ez]�trv9�fGO�S�m�]�P__�a�w�t�-V�����wI�>���en���RG��ゼ�����4#��Q_p���	Ǔ�᳜�U��uReLMU�gR>M��<������6���{?Cg1���.Gg���v��$�Zn�pu�L�u��M-�MMե����*��,�ʩ�Cb�C��@��&���˕Ԁ��9�Tac�P��5q���q��KO�nb���RZF�!�5%#�S
v��!�����!��K���n�p��`��2��R���)e.穥��V�)�?�U�F?o��瓒s�@3E��%C�0R��,�c�ｩ�#2��2N�@9j^(,,���=��'�
2�2y������ _<}��y��u/�g!�Ǯʱ�"�+�W�]5�&�|6~2���x��M���d*���j#,�8?מX[�+$�8���߃����Jƽ_���[�π��I久p��c�Ʒ���H&��`,���i|�σ���2p�筂"x!�P�16=�� �Ǯ]��iC�]Ն�|>�3�k�6
B������A���bjh�oi��m4��-&������x==]������RWX�`l����}��!5L��c�|Z�3茭f���n�{�>��i�����v�-vw�����ƅ��Fo �hs�ۺ�\�Uo�֠'��US}ea���'R6�8�TzNU}s���겶�T�'=]�{�p⇯�uBv��'Kgj�wy�=�N�?�8M������f��4�M�VcSS��ȣ?Nίj��D��-R����_�~H�j����o$��_+�rK��>���v��?��	�i^+���O�q�����鸻{�i��oG#���4�ί���8�:��<�q�H�|p��uc~s:�%��a~1����*\އ5��S/���8�SUS)6Xx��1$��.�g�r���[_}��wn�X^���9nbtb8�Z��|���[7o^�zyyi:��y���/ݼ��7���ha��к��pۙF�@ta���+ז/~~��/�|q�ʭ;_���|}s�ʍ[W��/߸)��H�޾��W�/\[Y�%y^�2��G'gfΝ�9��83<0��佴���`v�^�r��ͥs�'�BVSGOgU���ɉ�.���7w����۷o޸���_��/�ܾ�r��o�ܺyuqjz������D�)��x ����K��(�}-����&�q��~��'C=@c���QO��h0q94۾��ʛ߄Ztۖ�8������;�0O����I~^w���[k�q�؆�~.���=�0�|������.�m�OC��J�8�'�V�w���%����f���Y<�]6�d���2�F��?�~>>_�{=pA��;��d��$���_�����*"Y	�L�3@}u�a|��{�dO�	Mx�<����ܲ�QS�}�_�߫�;:����,.����a	,S��3��fK���H����6ה�����b�s�]�_2{����y��L�g�xg�5�@��~�u��H�P�߃w �$�}37��>74���������a�`]{պ�C#?�O�3���x,�\R)2x]�¹y.4V�*�l(#�'p,���.����s�2�٪�AdU��)S@���b�8�^^�T��o��Q��=�M�\�(17/Co��U��a&Ah7&��J4�ɲW&��4�z3��]��$�4D�%bI%�h !-��$��Hb AuF2V�96�5��!F�W��JH0Reb�Tv���DB��'y���<�{ �Âw� RI\E���cPI!B6��@��ƥa��4È�<U�E�$HJ�<2�4�E.�C���|?��j�ڠ����L~"��ҌLE�R��d�k�8�+L8�"���y�"2Hv�6���Z1x@Z+����#���
D%��ֶ�D�!Nh����􏡱�7,p(FW�Z5�k�����'�~? ����1$���XJIiB+4*R�l *H��V�R,h���r$6��O�I*�����'���"oB$�S,��j�ܙJ�*���Sɨ���P��I���	$0h�2��hT"��Hـ%xZ�w��]<�V���U
م�(o�~�p����S/���#�jy]�Lu�A�8�=I�0Z"c�X�[� ��62�i.�����Ӏ�%j�
�ҘQlD��-N�"lH-���.�]W'd�B 2I�_�'T�̢Z_����R}'M?�8@0R�O�N�x��T��2��-�����B9,X�*X��:�{������g���*�������	�I�4� @�D�Rg%M� �ΐ�qMIvZ�DVU<(��yqyP���+2�kA�	9�U�bA���2�,p�"I�
T�������	-0>�P�G\Z\FFQ�Ņd#k����)���e�dL � v��v��IE�Y�T�0n�C[l�$0E&e3�"�v?�3�@�R�A	)���"��r�S���)6��1��&�8��?d<q������idH
Ũm'�oS��C	�P������CA�x���U���7`I��pv(��6:�}G����o�6rv%�~d�����ĥ��$jD�P��#,/*��x	֟Jޏ��,7�ǡ	-~I4�łz�%�-o��F�I� g�!;k@�	��G+
���ɭ
�m*3�3i�B��	�RN��	��30!Gqy��]�=��
�# 	;emq�	ox ��dC�F�k��Rи���qb=�����Ef���L�
��x��~{�����PE��;
�	�)���6`R�{=��\<����R��_�v/����W������3�� �G`,J�����À���]�P��[�?����.C��ޢ@e�2�I�6j(!� NT��D:J��f�"��O��A:��42$�bӶ�� �Mվ�&��%�J%�S����Kj�2O"��߃D�hP�6M�ͤ� ��  /�$lT9YmMPd��j���` ��:�{��42$�b�� �[�8���}PǤ�F�Q�I��4����(�U�b�SRrR&����t9>�A�1�l⁸�|�9�$9�w��G�T0�ҬJjN�������T("V��L5�ϯC�'i�]�TمX?���c�����<���)_3Ã������"�$�Ck��"�=��;L�=�4%s^S/���DB�"�X�_k�����ȓk���)E�HQ2x�jC����� @�� �#TIf��F~�%�*O���%����ɮ���
���~w��ڜ)PZ�Rl�tRk�K� *N�(I��s���>�)�~5H��6���ֽ�0D�E ��k=��H^j��/C�P��&J���X�W�{�Ȼ�'yFR;/���C��I$��x�XE��2v��}�&�א�N%�S�������ǓH����D"������K��:�'�yz����g����@�C��EEb(9��w�*;��|4"��B��9���K �R�k�H7�`X� ���D;���`��(2��2�.!�J>��u���Db=�j	 �GU�&h�j�	����X��^�έD�$������߲�W�;��!jO�$&ŅZF��v'EChA1*~�Q�j�-�@`��� �(�:HQ
���.	΂�����F)%�UulP��V\�P aT�42$�b��8�*���Š6֓ �MM�BR0FM�*Pg�Wp!�'.�'��z��I��P�1�jmu���S�w`uՆ�Ĥ��}��d���G!U���L�X#�"I��5��͢Rb�4�w��{������A�����N {0$1 ɍz$ЄL(:��J�@Tk8�L�(��F)�P���aP�~�� �A�!�,4##���$$V�����w�H��_U�/��x��T@��{��*�Ȁ��.L1ԇ��u�˳<��R6�|��� ��7�<A!�R�F+�U'�v"O �<A �i6"�'��6$̀SD�xK~-��=�&��؃�#�+��	Q4'�Cv�x�Kr��f9E�'4J����w�b��5�ƚY^�Qێ�&gGT�6�biA�/�����)5��W�Jؕ�,� JZN�)��QgA:"I$�Cɼ�~D����#H�'4#B9,�Ȼ;�����)z,�F�<���V�����]X�c��k����x����U�c����X���'o��BBÏ�O"�~��"����� ���[�F#�p1`�2T4��R�8P���T�%o�����$��W�Bl>��2���v��ʵ��T"���rMq���ob �5���8�iTEB��^4��}a�'V��
�)��j࿵�~��'�a��*Z�G�U`�WN�,ˣؙ ԋ-��E������TREE  ������������������                              9             L       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OCHK    �Y            �     t  REFERENCE_LIST       dataset                                       dimension                                                                      y            �X�OHDR�                           ?      @ 4 4�      �w     X                   �8             shuffle            deflate            g                ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               P       DIMENSION_LIST                                    ��     �   ���OCHK    s�     �       P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               ��z�T�"            +��OHDRM(                �              �           ?      @ 4 4�      x     X                   �8             shuffle            deflate            �        �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   .WjOCHK            �P       DIMENSION_LIST                                    ��     �   �^&OCHKP        _FillValue       ?      @ 4 4�                               �43G�        �4             aVBOHDR (                �              �           ?      @ 4 4�      �     t                   �8             shuffle            deflate            F        �           ������������������������                  ?�,           ��6q  x^̝	<���'�Crt�]��	!�HB�"�R��(�$RIJ��ҡ�J�$�J!rTB)�IR���w������?�g�wf^׮�ٙݙ�@_(�ɜ7�N H:n(P8�9
S��H~�;eE����f�)�0��E��q��ݨ�*����v���
�ֻ�_��`�R�x�MoS���V�O)?�7�Ye�N�}p��9�T�d� ��7�e,����)d��bٕ#�P���n@���0y�;��0V
K-���u0&]~`�a t�~k��Vc0f�{+���`)��M� z�'����a�� x���o��l)��X	,�: �!_`���C>��xsX�p�)~I,}���b�)u ��)�Ǖ��6�����s �L��Mp��G���p��0WRF�1�3����M>�����"k�����4	ez/3�tF��Y�I������1��7�MBu�%������x.})�G�N}pK�@�h�a�B
�1	���$�=�[�UxJdI�D>.t�{�eZ�r-��p��E�r0���ZJÙf�q?���؊��r8OhJ����܄z�!b��F�6����?�2����F��T�Zu�U�3����f+]^�va��"$kҔ���ݮWj�RJ��%>��$�r{��i��N��ӌ������U5mk��=������<���߇�ޭ��ol5�GF����)yڳ��t���u ��X� *'HvV�Z�:�bk�N��ܜhH�s@���ƠA;Ju��������탙_'c��%�,�A�w��O���pn�q7g����,�)���(�T�
���Y���^؁� $ېY����ڇlx�����L���r�N R�n;~!����� $"骃��!~	���P,��H��ũJ�z��r��#��6�t�����Bû��:+��6.�d{ �aМ8���C~(7��c��Ղ��K��N2SHqT��u����N'��{�e�6���?Z\�J�[o���#̜Q�1ܡY�2[��x0cӀ����&����q�ؕ#�����~\�}���'� ���o��5t�2��y��y���@����M����ڒ*��No�Y�p������]y�������.Y��NQ�x57=���Sm2�/T����%fB�I�ӧ;U�Nw���>Z���T�5��
es�Ī�7�]���g�<�&b�����*?>�٤<)靪��s�"���k_��Zn����aˮ����YyA���f���K�O.��Uչ4e�jߕ�jNd����r���w;����U��Z?Mg��է��-tT}d|Q�x�]�y���T�|��e��|s't�V|�A�P��V��������V�t��Ə���yo�߶m�|�}�[nw_�1+rc���v��n�1�#�D)����ɈV�,�>�-��~5�<l���-A�?���b�4qt���=t�"�"/p���$������l�D�濤�^���1��#8��L$������'{�����6إӨ7�A��<�8�V"�ڻ�E�;��I`]j�3r;��ɓaOݰm,� ����O� �[��k��Cx�ZF�����PF@iʈZ�m��i��?C4�d/�Vh/�ZҊt"�0�"�&t�� ���F��2&�6"x���xB&�n���J��I{s�"��F��)�����Uន+FХ�iX��K?F���h)��O�1y��8��m!�^4FS{KR�o��i�g�!Cp�{g���򖦄,;���2IA�M�H��A6˵7���=f�,`kqh!ג%�'��%$ȥ��H���8�Q�\E>!x
��\o���[�2@����4Bh�rK�ߙ��*����A��W�'~�| �2襒"�`��n7{ѩ�l<t�]���g��Eu�Qz�nL|�]��x�T�`��g���я��>�r�����h��kG�Z~�meF���k��f�-M��3kf����Q��Ė�_���jޡ�w3���W�<��M 9�6�'�a��d��9�?�:��eqs��F#/�*}^g�/�S$������G���&����	��9��{�KK�3�v�>��t��p�>���lUݽi��4���|9��.�2mv�+/��<����dLC��"�L��\OLO�|�s'd�W٨.���l�����k�$H�_Z�􃕲�O�ZY��6�Z��+���5��]�Jix������e�y��i�{���x0'����Q܇xJ�D�w�i4<Ӣ)s1�X�ѣ_*#��|P�|El�p+woE����G��ڽ*�75B�l�9-�͡���mW��ṼB�{�x�@�9��?}+%ֽh�� �6�Ӿ����-���W�H��K��I�s��Y��/V?�������S�6}�n�R>ê[��3��K�{��[���:N�<�7!o����B4�͒�%��;��$������d��
2˒3��9^�'D7���� ̍��9@i{��Ɔ�3��N-36�Wq�	�zC�ZU��**G KRR�H� 5����ھ�D��r(�	�G�hB���-4c�|xd"$��M�h�-�w�!i�-�*Qd�4��Q1srΙ "��]x;6 �Q���<�Ck �>��!�bpA���!���x��@^Ul��.R\�1:�%';%Ζ���4��h�85�����Wo)l�L� ��V�>f�"�R��� �Јk>]\ʱ�7a�$�5 J�O�����Zr�.$`_��w��m0����~2[�Cy����4�j��6R'C�3	Woo������ֽ�G���u��Bd���9�X
k<W� �Os����M,"��*�d�H!�A��i_*rZ��Fx\�J5�>�b/Q�+����Zz��D��|\8����rw�h�n>q�WP���Һ�W7w^׮�~k�\���r����buk����v*��E���^w���
���ً���y�˓���c~N)/^���P5�]G��W��6�Il�z$
x��:��^\X��v��%����
����w��>o��¥�'��o��X��0��YcA��.ɌS�<��OM��"��8���g]-i�ޙ54��N���Ҋ�]�ǃG�W��7ײ栶�ͮ�Ǌ��Dg����&�X�L���=�v%��4~�Ԕ!��De/~9����?��p��rV{<��
��M�SY����%~PD%g���A�%��7� ��{AGxG�b5�f;�?�����k�m	)h�Ga�G�7��T�W֔�ۊ�"l����@XY,�Y~�lѢJ< ���"(��u�^�1����jl��)d����'sh�V �fM�C��e���t�b7�N0��$�*9G&ତ|��(�B��a��j�O4:�u?~61�3:���Q_.L/|/tYog\��a>��rtĹË.N�lh����`Ho/ܖ-e�)�Gڑ�6̩۰^�^(�q�����뼹@L��[�%ǷS�e��:[qJ����s�i�w��5��=�& U?HV��ҹ�B�ekD4�^?p�q��*��=�{N:|\ꣶ���2'~=�I��_�f�@�X���+r�q5Չk�����߫�%r'����#��ڿ��������b2�3��Oz�7(e������f||��s΃�Y���s�_|�i�"��gCGx��:�1Ft��5��3��d��)x?�g��e�;_\�L�<�{��Ճ�k�H�K򏎘����O����uD�=Q<7p����_��M�����l������kӣ;��U���z��������-�������I#���n�_�y����4#�gR��
�{4N�#������u(��ynvNϬ��S�4�x}ԥ�<d�&����Z;�Eͯo����q����ꚟ�dCfe}/���qC0X��F����y��E�l��Ν�9�0�B~� ���/3��,�1u��S��.t5[�j���tfM}��)�]Ϭ����N.��j�zzU�>��>~`�%,���	�-	�U�ϔ#E��@)��Y��y=ܯgZ���u|���u;:���?��}�G/0a�!BZ��/;�i�S��CZ'.J�Pq9���V,g{���:�:a�q��yz:,.�j;b1�mĤtfPP(R`��C�����*�Q��&�Hy�g��1�@,��o �Ik��B<��˧H!i �Q�g���;=Ķ�A�$Z�bd�	��x���?�gX�t0Q5��+^D�M���3d4�KH(�?�a3a�Y��Qj��� 8��"��Ws�U��-�.bQ}l���p�o����O ���_�K�%�y]�~|����ǝ��1F���󤤂iB\�wƇ�0=�!RR�2�c�s�P`����!�,a��C�$�O��PK���x���'k�\�]bW2��d���X����ͻoz%j�:#�i�}��F�<�<�v7<7?8�E��%m�͗<3/���<�-8�9!�UC|���O=���H�k�:7��(I��۫��fu_��[����7�ۆ���g����������=O�}��>6�,�L+oh>�߶��y�%��	�{�~���@�Ϩ�r��l喗/��L��mO�W������*�fQ�����7CG��
��O�q���|7���3,������Z�?>Z�y�}*��B��M�ɂ�N4-r����mT<���~+@/���9����ɶ��6��a+��@�bVL�4pI���d������j�+��b�&��˴�kpr�����j�k[Nr�G�Q
��II��X�!�� ~C$��N����-��	�cYQA�]> 	��l�\	:���0=	�ؑ���hS	DOup��UG��׎a�>pD�����L (B>�x>J���"4�3'�,)��2�]$b�Io>qΥ�<9O�Vt�u��73{]$����i�['�Nt7s5��z��՗���e�;��w_��4]c�,��������;����%�	�y1_���ǩ=B�DB�h�>�-�����D�7����pSWx��l����� �Q3��RG*��4�9�	�c5�� l��s>�� N�u7���3�㩙��,�R�`�M�� ��\�O��X7V�Zc�0�Ik4�hp-�M�^t%�.0ŗԡg-�5�4�6l��� ���/v\�I ���?� ^1,#3��Yv��S#y;���'�U%�ə�K�=���m�����>�b>��af�r���v��jH�g'\�;eu���/K#Ǡ)������>�n7mP|@Kx��[���,S�k��~��=hT)����2\�{j)]2D����+�����
�L?_��J��1-��E&����Z�IZ��/���������ާ���횚;G��X�s�X��U�/yE�AWb���9
~������vVJm�����2R�6�(1��ǘ��NQa�C�����"<G���5Ҿ�S��v�O2��
`	5N:�	�ˢчdp6n ���̌M2V\�&:L���u�	H�ïNQB��R?�Gy��%2���'%����R�~�pdw�����'"c���;q��t{<}_TX���r�Fi�Q���3֫��{���op�?i���u�Ϗ����fJט̪��_o���(+}�i+�1¼�T��6|����9��[�����T��n97+Y�`�����!/b�;d;�l���:r���kĭ͆&����Ӻy�'�1���)9 S�A�7���D[k��+*�����"�u�S��5s�u]��g�cYy���A�|���U�Ϯ�8[��9qʷ3�[m.
/\�69��ϕR��A/��S8e�1�նuy�����go��NY��юͻ,\�m��S�������e�5�r�ڃ�e�C:���`S�,|M+�$���x�×��,�u�s��	�V��p�Ktj}Οf�L���5�3���Y"��3Z݋����6#�yΖ����h������b�h��>̱O�ﱒ���WWW��d�xD]��g���h��$U٥f�<�4G�nYT�R��!|z�ī������:T�|�������Zb[`|�����y��D8�����}�x��޸%v�V�;T4�,>v�k�[��{DGL�ɍ�ؕ�a��@�Y��iZ�����#R!�ue����|�Q��1YOo_t�p��4�dD�����S��K�͏������-
wW���0Ȳ͹g�?�l|x���֓�.�ޚl�n�2S��!ƭ�o�۶���`��+���Ҡ��Gš+P�V!P��� �9C��Z�
�L��1������b�#L��S*QZ�R�~�I����J��[�u�[�~�Z4�r(X�^��1Jwߜ-H�<��d�����`Z�;x!Twa�,x��_�.�}5�<q�������;���b+�G������z����4Sb���A�28A"�J��D�	��r�꾈7�'�ar�M?�}��q��{.�׼`�0�z}��wG���%J4<�D�Q(%>4���8�nd�h��Pj�C
�j���NR5�]���H�H� tH�	}���90�c,��x:l؀�b W���!�RSf-.�qhm��/��ߵ������9.������4����m'�d)OngI�Ѥ�^~�	�WI���s��Z{�Ј��-r-D.A�����w�����ː[lb}>a�#�ĴǐE���	 ;�W��j�[�����;6nŅ��+�����	�U}5�B�+���*��4"(���2*���� �z�%�yLB�H�ݒ.�4�PL�����N�����&H�6��l#wtl"t߻ѝzg(s�C28�T<���M��O�� ���ٲ�c� �"7��0:��˜�� ��0_BD�����|Q�������}#�8��'���D��
�$c��� �8�'�xHі$�����>2�_s���Kg�'8����"�w���H��4�#�p��>>����_�H'�,��zW�s�<Q�t���{cx��زa�ֻ�;��>�m:� �e� �8��W����O_Ӝ�s�|�*њ?�w��hl�r����O���=��س��F|d΍�¬���$VȘn�56pj�z��v�N�o$�^O9\�I��q��-�ª�y⥟K�'b��.չ|ܘ/�&��~ұ?���}���[4D���M�2�Ѯ7ૹ%�S���y!"��"�N
��θr�f{�#
�A�Jĵ�&$^���h:��dG�
1�+���u�����@�_q�y���&'��]yxH�Q����w��o˃�O�������w�?����>����^﹋K�ןc�5pp�#p����x���pC��8�Q�Q�ru�QTA�)("�,���O���M/z��Ў�e/�F��WI|�PM/��˜E�7U��|}�b8���ǃnz�e>��"$xJ���9�|�F�����=s8<=��qo*���e,�t��nt��O�?@@�"�Z�̙Kar�!q���qy�M ���ڋ��K��G�;��#]�����@x��w���}@F�n���˙�s�E�п��-ř��w�Ϯ#���">��"�Ȭ�BܒӨ�9|�<�b>_����F�2�ܨ��(���b5N,!��o36�\����? w�\I���̨W��6��?�P�>�}����&�$��	�̔ҩF���h)���^2@8��Wp�:*�#�(A_��#s�:����#�P� ܒ?�MxF�V�]���!&�Z�=(L!�f�z4,�YR�*�mv94菙��wr��=)�c�q>,o�uU#��/Q�T��hl������YE��Ǟ�Yx>mɞ���`(�W7��dL�K,r㯯W�e+8,�����˳�3��퇹H�����@�֑�v(�Z��k+�����ٓ>ݵ!e�5�+�M���}m��MKr����u�yl�]��ǎ q���I���.q���x�X�� ����#|51[Kl�����**y�[�����Ov�>�q�I7�v��`�D��(�������
V�N$�,fC�Fү	O�-�E����'�ۄ�4��8�}}�=��>��o�h �������d���%/P��wy��.d�Z����*�X�t��p�e�^�"��.'W-���~�%�Z�ti�ヶ���ݣ-���k��ÿV��g����mE�
S�\z�0�'O. ��wq��=�ct��M�v�+#(n��r��7�:_��0��\��g��U2+���ف�~���{��5M��桼����,t�����dz���u�Zr��G������W�z��^��n\RX��QZ�b�-���4���	�j�s_���v�+wk<���;mf.S�+�x]<)��.��g��L��x���O����si?w�r}���'��⏝pN��.����������ӷ�yR[`ĭ�^ź�����1I���r���};�rP�����!7G�/H0���qg��١�v7��'�iY��%�3�����{箼�Qj�r���������V�S.S��w�����޷�r�.�>j�5���/�6�N�iNٔlmw!��ָ����s���:;�u���%���5�K�rPh���\�+���=j�٬��o������+Hu�y{�~p�-��9&�gM��?"V>��|�d��sW��j,��.5����'���]u5w7��fL�A���A����u�7@�=[��q;v}5��e�ߌ�HO��Q
���{���χ��m��i�W|=�/���َ�)�/�ڿd1*~�%��B�� ��ј����pċ��P#[6�|P�o��.�Vb�L���/i�,���Kӧ���5c�l�QD.��i %s:�{�rF@b	�z6�az,�B���g��giO�2��%��2^ ���DjaJE�L��}A#͍T�eLHm��9�H|�^���?7HGD��+!0|�5����
b��{M`M��{J(�K=��S����ћ��k� O ��fa
��p!�,T��ɝ�$�p�lN��op7%�;�L�)!Ow�Ln"^�F�����f��0r���ӆk�B�'Kl#N��^UL�T�I��B�u`x�d�����.���G��P�e3O5m�@��b!v�V�(A'5�;���@.�[L�[!���5&n�"w�},q+����y��{�*NzyeX��C�#yǏ<�����&IU�y��,7��i�����q�(~A\5�E�Q0�p{s����Nc<��73ϴR�ުV_ϣ;�J8+\�0�W�����+�� �j�i�����8El�J���
�#����H��Y}<��ޘ�=�w>:������n�k�]P�ʢE����3�7K��u6�o�6)�<�]�R,jx^o�Ӑ�Q��t����CVpA�Db������V��_{��b�zjv哰�\�C�����3;�v�lS}�Q���$g6��ne2�[M��eæ���/������wcTm�/�|�7u����Weozں,�JP������M�z}bl��z�_xd3R�5����y��0���i
7�����
����*��4��I�q2�i˶���uC-O٨�������ʝr�"��s
MMN��~p�霼�kz��R�͐u<gl�|�^	��f���ذ�T�����[㋯G��xGj��j����L~î��`�ˣ9��/��d,�%ph�����ip��}	�eə�h�����E{� �F������sc��EF'��\��8��X�!k�*�kU���y�#�U���ut�#w�oI�,R%>�xkM��Z�����樻�If���bj*�Q@v_z�?�d7w`[�`t��U �!s�N�A���y�~a��:�h������3P�M�"$��< ��C.D�O�a���(��˭����H�>���U=�+�JsD���>83��{�㠓m�̀oqҨ�f��a��3�s5l~ ЍG�x]�O.0�f�$aϮР]�s���U9		RiJ��H�S�0D��������$���D/\��4�j��ARG<��C��'��r�I�H���#�j�2�^b!2}��\�7��M[(�4'[)`��7��/�d�I!������B��_��q�N��>���ERH��l�Z5�9�����x_9�J�,ފ�8ˠ��+>Ϫh8Т�d1�k�����%�xj�:�ҳG�o�>ت��o��@�uO�r�T��̆l�3{֊�h�)T���o~[>'O�z}3י����M�r����ㆫ�jj�����/��|�dO��!�?�(c�k��/C����|��w��s�N�=#)�%ZI.�m�h�������gIu�h�w0�L�3iV?���e�8�O��6_��\�D��w�<:�%Ć�\�6�r��8n�n3��3w8M�!�\��G������؟#l�I`X��ܟ+8���a����S����,s�1Ӡ�����^��hcp�e3�[Ys-P�r�/!���C���H�h��p����m��ax����@XY,�Y~����$< �|�c(�ٍ"��*�`��j3B�۠yG2�fh�Phִ84H[p?X0N!vC��S��T�92g%僇`D���=V��x��A���y���4��ra�x�{��z;�~\��vx��g�M�h�9�Т�-=4���^�-[ʆS4r��#'m8�S�a�4�xY�$7�%�ďI|35_6:-^R��^�x���e���jS�*�=2�6��\���G�V�I>�nk1�����c�.	h�+|�7:��n�S���ʢ��A^a[�Wl�v�>s�9�La����ڐ��g���3�ɾ��Mʿ������*�q��d=}]m����6�����T{�B��*��&y��>.���7{&�k<���f��c[�dV7S{��5������r�(�xw@GyN��Ѽ1���'����,ڽ�e���U�r��Z�kY��*������SO�Ρ��
Vp%�O̝���aiT9k��8GW��!��%�e>7��P���d1e��ږU�/�6�t��n��{xO�����Nr�F�g~�o�H���ˊ��W�uu������\]��^9��*cQ��u��!Su3�I7�]��G�^h��77��Vm���x��2�����*Dںn�U�lQ�ըxʩM͑��t,�C�~0ۖ{���٭k�{�����ӈg�҃�t��?Xqu�Wh�рٺ;���ހ^U�𺏟X�x|�0AhGMm��b2��)DCQe�1�FJ�=D�v>ZyG}��x+�&}Z�˖�\�`W��p/�A���A��\���X�������ظ΂׼���[_ڎYsD[�_P��Qp�����~�8ߤؔmĤS�HKO�f��C���u4T�R��f. ;�r��!���'b����$NZ�\�7��xy)d���ۗV�	�SƢXTM�U(F�^��:�W(z��y�%nKUK��E�(Ф?�=AF�����#���E�����5�k�{Oў9�F��o�1v��xڇ�)����w������ �<������9�|6��|e��;��'ָ��_�kut<hB\�wƇ�'""#%@��}:I
��K̃�}��I
H��l�?f�Q�֖���M��s&�֫)������f~��w��f�9ϷE���|�~�j{H��A:�|��8��8�0�i���_��hՠ�2��`����lq����3���8Rm����8vi��X��������"�T�Bruѷ�۫�+s��uS�X�JU�z9q�g���Q���������\���R�k;�wk�*��~xg����s�\uc��CV�-vɹ6稀܂)��?|N�<�P���C�!���r���(���'c;�W��+ߛF��l����OM�M�F~XF��x4ᢰRV�^��sB��ӱb��@��а��{�]1�?�[x�c]A�Lj�H�EW��F�D�k]z����kO�h
d)�=�C�(���5�L`	m�Ag�ɥ��A�����v�6�NP�О�*]�.�,f����]67F����cR ]l;��p��t�&t�	T�����P�`�>c���!0�kކ׾c� ?(�/��4��c�}��>��h��g!��f���cI͒b=��!<e[�9t�i#T�w��*/rD�Ty�
ml=$6^���� ��?^���!Ε�[�=-'T��{���F���K�X [�K-ѩ�Ml�ެ�<��r�oQA>�ZpB���$n.��f�uc&k}o�X�U�½=�yN�qvHg�rk����UH��,h�0�f�X\�C�&��7�\�ȄL��
m_8X�:�?㇑�� �14i"9����:��_��D�	ǰr-��	XZ G�&� ��eO�i�J*)�s*� |�C�X\Bg���~�x.N����bC�����W��2|U�Iv�A��^s����"��X>qӫ1�U�+�о���st�����_G]h������ְ�W!���,[;lG½��ҟ:�6��ϊA�}�~���\lSyД+9�j�� �o7��ӷ�tM����BᎵ��_&6�[�O}g}����f�E����$FT,��R5Fz�piI�2�����k�G�]Q\�����)2�~#�F���3���w\c�����iץm���� :z�I�z|�B����ACP��|�G�8܏�k�z\Sv��x�������I��`v�r�H��w��wp�ޏb�㈓K��RɤnȞ���8w���m�=2��~���]0Pڿ��1���� �}Qai������U���M����k�^prḈ�!�K���<}��J����F
&��bZ:5�;��4{��Y���Ѯ%S���o�o�H��٦I��&���m\b�}�2����5i1��uw/��Μ1tb˻��}f���q�${�B*
7_;�>�2���RY��#����h��wz�Q"6u�d���umq:������Ioy�sl��X���ܷq����;�����A�ї�J�-�ޱ��	I^wi)����U����{���̛�I���쾦�9��L��k1��-	�ɸ�q"U�w���@D)��n{��1/���*�)���{�N���!U�+�$b����$F�_Aj�K5��Ş�f$��ؒ�������\YTI,P�e`=O�X�-�����,��w���f�.u=}՝R�r�}���w���XW�ΐ]7ֺ.�q�ư��wLV�X3�+kdw����9zq)q�/�4����`6GZ���ĺ�\�w�9�;̾29����z�[����G
����9���iM/�b�4�V�=�K �Zb�ꊦ��˕�]�;7��f庹�"�W9"�$��Rb�鉟��_m���{��@��CcS���͎<�.=�Uw���D��ecoIܾ�slڂC+	(heڞ�Z0���UzQ��M�W7�orj��;�m�t�D�uo4~�`Tu���Ͳ5J����+Hΐv�nص�"���~� ����]�����򒽃*��?��(�|l�����Mվ�K�=��Q 6��Y��MU� ��t��K�z�.ox��)0����]5!H.h�TM)9���T?Ӌ���!�	fA��*P�w��L�b�����+�Cta�ʱ(�����p��&���c��<gi��{��Ӯ�W�wt=�Y�D3��Ndu�R�CӰ�!�s��G��R���z9|\����$U�X��`��AH��,I��K/��֊�Æ���/R`#��OGG�9���' ����ܡ�	/�_R���엖Ŋ)���|�V����;�.�u�n�{�:�8�����! �]v+�U����y/��o؝�����OaZ��nC����<��I��[lt�����hX������ v��p�#��O�������M�����U��$b�B_�P������_ep�fA�b^PF����U����0�I(i�ۡ[�%�!�щ)l��~�Ć����E�P�6
@w�!���zC���4�h��aвh���C?�m�M�q*ǔ|��֭^�v�0G{�!�ub�,A+���0D���G{ ���ø�� �y'�Oe�^�[��dqy"%GN��ah�.�vct�5E��\�ً��ɚ�}�PT��p�I���>�l�!�s�PDm�Gv��atVѐ���2���nr���M{�|�v��m�3�t���"�ȉ�la�5ʺ����㩫�-c��^3s@��i�Q��scV��V������eQY�$�ŀ�-G/V-}�ۈ볍���г}w��I�W3����]������h�+�%C_�	;5Ԙ]��a���2���L�S�-�s�6m�3lW:?��za�ڛ�k�|�����^6�NO��(N�r�����oz����.+l6Z�4ҫ���d�+�l��>�y�=E[�xz��Wa����"f�vKq����Ė�ہ�.X��U󕹫��2Y \t��_�ĀS3��qm������������� ���z�N�9��bN��2ps�\����)("�,5v��?#㡛^����'�^Ѝ���>��^ a�9�0o��oe���p�*%���6%�20<aEH��c{s�8���)�N�3��Ch��w�l��ml	�l�AGA���E�n�`SË�����#��x-b7	�`�ŞL0Y��kR?uG��t�jb�C����A�m,�a(>��.�qb��'�c=��z���t��� j���$�e빐k��02y�|���	b��Ө�7�^�s=>�e; �̐3� �P�n�1!tu	�ߝ�I����8��!W�--p��0WRF�1�3����>��>qC����+ZhQ�I((�x�)���fE'��X�,&���\�
��H+J�-�	آK<�����#P� ܒ?Щ�h�a��1	�2_�qO4�֣�a��ȒT�|6�Ŋ���JNoW�{y��
���Q"��D#���EsTK�zoGֈ��eG��<X>nܨc�l5�f�.��b�GLT*d������2L@�?Wu�ym��y�����yӾ!��1FTi% ;L��(ڧή�^F��t�%����	3=w{L�r��������\�|M}KRڙ�<Ww.�{��y����ic�R�w���;��7>Y����F�!�����> ��L���|��f�w�,za{�VWWWr]�_�Jh���t#hG)��<��L ���}0sX����)d���I�x���)��HPF�~Ñ<`��~/)Ae��-�:��f@�6��tMz+�.���^� �	�ú[6~��7�Va.�m��[<�od��Y��|G��$,�M -�����&�#��N���!|}�����J��s��^l<��V���'�<�n�jN�ҷ��]�:<��t��8�1.W��Q͑��Q�9�	���^��s�q�v�ظb@�_��CSt�i�?����C�sJ�t����S�n,�$�5#����j�z�E�F���a�����R����^�i<v��EO�/����b�)�kČ�{��zzg��7�N�9C�j��:��¢�7�|!*ow?4�������QB�ur/Q�P".g���[�	~7=g%��,�?9%Fj����G�8�5���]ly�'GD�l�f�����u=��ō����6N�Z�� ���Q��7�4�O����a_�K���g��t���r�3l��-k6��|�q���#|U�7O���1��t��E�y��~�)������g]���ѹbj�S���S~�g��{���i~JB�UᢜTMc+혗~U�_ħ�v���:X�|V��ټM|��=���S�-$}���^�O�}k~%<60|�]{��)鮙-�l�_·!",֥�\�r_���0�Ԣ��W	(�/����KK��H&��ѧN��)�`�!�k*��"F��;���J������S7L��EL>|?�ŵ�Q��D�<[��0v��=Q�cd��E,|)��u N*�I�i>(���Kg�Ȅ�G�6�����n��,�9��χ��"���WE@{�>ٿ�4�_}��p��2�XW��Z������N4�d/�Vh/�ZҊt"�0%���	&����;��@�1!��e�>qw��a��L+3�Nkp�#��N/�
1h����鯄��=Ǵ2F_g�O��㿨d�>���&�U��L��v��9 �"�O��J*1�U�1Q,=�1�Ӈ��t'�g4%D��S7��m��|)/�b0��ڛc�=���s$���t%Kl#N袳2���W�"���l!��t�m�ɂ��Z�d�3��_��˶8`L�����~_��f��֒.�u�	��hchu��݉\���.W^�8�s�n/{) s��h]B�^@��+a�ޟ"�7�;�L炄r��owW���b�L��y[ܟ���m�:�YKN�z��kk�+uf>�Ќ�ٯ��(��Kw���+F&m�hU�W�qO�:L�î���g��}�Ň~�o�<�p�H�B�k'~$�>����C�i���+�T�G��)�2֨��=-s|��Y�/u3W_� ��6��b�z>�V|!f<$l�! y=F��@N��M������FgJ��%	^�����BÒ)�'�}�]�*�2r����)�~�dV��>:n�p����g��+.��N���W�tl���d�I�^c�4���O&A������?���ىڵ�g˥�\�[񳡕���j:�J��[�D.֝�>)c>l��I���C�!�-"���[Wzx	���|P��s�	�7��|�����&Z�ơ��n����*�x�gX9aY֌�˵f�Z-��,g���6���W`���4z"-2
�ĵ��}��gm����𥉉���/�}�3և]	�O����Bs�Yҿ���& ��$������d��
2˒3��9^�'��C�����h`nȦo(m/Cf�X�0��B����B\ŌC����Ъ�V��_��0�΀ܸ�-iu?�ƞ�JF]�G�h�Rk�)ކ��xd"$�9�/�P���1�o�I�JRVq}�c|�����Rr�| �0W�3��]�(\T���| G����F��ܟ/�C�77�F�����s@SO��{zJ&�d��F�a�=����n'}p�Y�p�Dp�m�~!�^�b�C�[6w3 � �`Ɩ�f��d�!�1I�-�)h�m���n�愱���b�6�r%C�� i��$'-�z��T�ثu�SL/HyՎOB�J���L��9�	��2t4��5�g�8���4L9�SI� �v�`?1�r��a�!GX6)���H6�e���X������E�lP�Y@xH
e�E�HJ����*�����p�˵�+�a_TrQ" i���/x���]ؑ��~c}̔PYk��"��
W�Yl�����a���)�+�mRw�H�p�6�j��rۃ������'��2�� �bHސI���Ew?|	�B��><>L��ի�G�O�y������^���"�>w��c�kM��1� �An���4�X��[0�nlY�����>�ؗ.���:(?6+�b�5�w�.�����דA��SԻ��+��:EDFZ(~om�H_^�f�N�*�L����&l	Ơ�x}$ꏌL_�G��9	��=��s�YM�SY���L$~PD%g���A�%�� ֡{A�F��s�l. A3D<��%�� rә��zcF���|>��öb�m�a�m���������e� 4�%�PB�[O��ҍ�?M�V���4!K�m�� 2�fh�Phִ84H[p?X0N!vC���<T�92g%僇`D���=V��x��A���y���4��ra�x��{��z;�'��!��#��Y p#�Ülh����`bo/ܖ-e�)�Gڑ�6̩۰^�^Ȕ���Y<�.���m2�כ�8�~y��U�r�Z.[��.->|����̶ҳ�W�_{���%���"�[�"���Ζ�TYp���O�����y�rY�位QQ��=��n��C|k�A�R���y^�e�g�����Д�6Xg��\$�`;�_�����rnK���v��`,*7�7!�n�R�娅K�W7�n�W�t'E�hk���L����/�7[�1m����6Yq˙xʝ08yk��������ǚ�L��ğ�R���ࢥ�Kb̞�^��3g��=ON%�H��愘�ݬ�Bk�j�[�o?���Z�+�� ��U+�~��NJ��v|WNa���L���'�d';�i���l�Lr�����'��SK���=���ĸ����Q��dL���,6u��
���3���j�}w�aO���2u�G޵,�t�ٵ�A����5jT��,�?��!=�ƚ��W��z����V��~���$ïe��\��c�G��'d~/SXV�8.,���o����v�^/xy�|(U#8�Ы��^��s �˒�G�	�-O���C�8�h(��8��Hi����*�a���]�j��:1��O(1hkO�RJ��d���dq�� �C��&~����"p2~L>�	�U�9���c|�ߍ������v�tp��?9����X�a��*��4����9�~��g�)�A��� e�#�%Հ
b��[�`iMs�|��婤��$��ҡF��
�N��V��D�P�,�"�u�P���Kܖ& �&�l���Q�I{&�l"|		e�Gvp1�t���>��k�93׽�<i�J��6�Λ��5��9�y�����Ԗ�t�����/�����[��H���w�CO�q3\�D��h��C;t�
�pg|8�p""2R��x�1�#XdbyZ?0��>x"$) }�7쯀��e|�95E�H밥g�s��T��V�Ξ�}��U�ҝ�m
����B�}�9�qI��,�� ���7�J��V����Hϛ�ԁ�G�v�Z���,o���������b����m��#;��X(�W+���
� ��}}6����4yRwh�l�ܵ��JH�׉tq��M��U�&��]t����&��נ��Y����u`�9����G�Yi�,�����O��7_�':���n�UG�{~I���t�U���F�Q��7�<�U�F�K��i���:r����C��#�@E��x4�_Xz�K3�D���̶��6��a+��@�bVL�4��b]A�Lj�H���8HI�E�������|�CB1���zYp}G:������GI�����58e*>.f�`<4�qu��{ِ;���	d�t
�׎�q�I.��h�p����3C�j���/3	�,%&��܁�j�L�����!`?��c�<�e@��MVt1�|#�����@+���᫨a�K��Uqn���13�mE'��;����<�ę�����ꔸE}nl2��s��z��^��J��p�Nt�%U�x�5�݌�]�$�9��g*ތ��K�o5X��f��z�����߱4��K�l�%�C���2^����1_>��(�K�b�Y���!��i����7 ��뱘bLi�&]����{�Ӛ����6 ,���o�a�>���%XZ��%�W"Q$��>��';��C�^Y �F��+��#z����.�a�{x���_�ύ`�` ���_� +�;�܎�x_�hNs��[����?7���wQ\XX��ً�gw��x_Vl�w������N��K�d����n׸�nH�ΘY�#�g)�6b�������w��(R�1�{��K�s��D����^��J=�K"|AX����T�a�|�_�z�͝$�V=�٥��e��i����.��C�W�>���1����|v=w�P��M^z��|A�r����\���4v~,W���̋����	��>H/��DqYc����Pbk��#��䏫L[�wm"���.e@9R5f��4ұn����Z����@w �KQP%3�o��74����D���߇���1��'u�hB>xy�u���(ٓ}��v�D�M8��GFP����J�ם8�#�=��/*,mt9��ۧ��c�j�<�u��48y������ʜIk�<H�~w-}��`��#|�-���鹁��q�#5��|t"Z�1����w��M5>�����1ɡ����y9�V�v��2�����o��2���7�O=t^H�޲�$3�j�����.�f|��br��P�s�m��
����<� ���oد�v������<��4�w:r���A�g�Ҕ��>k�çuw���Q��:��ls���Pr��W�'JZ�?/iz����z3�ȃq�W���F���|��ԣ^fƈ�����2�t����{x,����d�H�JF�"�%�H	)M!!m��(RQD��hH��h�ѐD���$��Rf���^�E�������=뾮s^�\�};�9�:�>��6�̺��a�_V����ӹل�cH��-/6Z/Ѓ�X�����-�����,*�	�[rM1�uc�#�ɻ��6I�zI>W�![[��	`�B>�h���r��7|.�9����Q�*����X��j���T.��dƽ�jm��ڍ�����Ly��� �}��s�Mz�#wg�����gO:�:cQ�G����O������t�}���ɖ/�LڥM�c�������0��q{^��Ƕ��dI����T/�7: �Z�yے�>��M߻�H��u;����\A���0���f�v��ŵ�Jz�}A�׫�$^J�يU�<;Bϯ��{s�����#7�=&=���Q�q]oy��i_.�0gD_N�S�g�~o�( -��������J>�� ��!�8�a.�PR�*K��܎�D��<�|xF����Dv�����0e `u�ؾ�aK~�
�9BF�i�P�g�Y����0�v��0�6�zWfX��ȼ��R�Āy�ͺޕ+��z�k�����	z}A��*�Ba<����D-0ܕ�*��t��P!,GSYGV_x��23�y?�t��~`�%r�ݎ�#�>;���� �`	aNDq�����a:�C g��&bXj���j��/����4lR8	���� �ȳQ�Hq��y��ƏujC���z4"��}�G�Td} ָШh�VIr7����Ţc������wf/OqW!�p+��o��g3����C-"�D�Z�,�plMS	9x�<����c���. Ic4�e5m���q�4�M���(n��� ��@�)���O;<g<x/��D$������G�{2Ogܽ�l�D��Z诀@&*0�h;Za�<(x�dڸ,#�h��a�qq��RG�`vȑpᡀ�(4��c�]x��A�̆���o6L#�L$�y�cJK�Ҽ�<2=���E����W���3��p�A䨒7J��aN)g ����ِ�61����%�p�d�	����5�̱������<t�md�=�Rv!�>w<(�v7$����*`<�a�
�ȶ?hw��"�u;[q����G���@~xa�m��
�.��Z �o����ߟWYDֺ,�]����?rܒ�S��X�1���m�ޕ��O~�Ӿ~⃇�r��z�P���-^Z➦���<>�q�hd۷��9[LTA�i�����3�D<>��>4"i�߲����߯�n�Z.�d�9֝����2#�pAn�ɦ*��cOlZ�볳���ǟ��%4<w�9m۴<�a^�eK�l]s�ҡ�WG���i�.��=}��ǵ��G�tŏ�,�'eg������W���MOI8s����pa�g�w�Y ��Mr7�}q+;�.��[H�tYD�����.C��
rNϲh�P{~�����U�S����H�#�y�V���w����38�Ȼ�lz��E�	"c�U�*t�$�ed�(vQ��3����?���8�|߰�%T#w��?OXM�0��"ԛ�`G�~�0,�AA��� NS<�'�[��L�1�Y0cęŮ��9����z��lHl����R��� �>�?ө� �H�Ng�D�X�\��l��),�h������	��[@����! �1g)lj/�z��9���6�������&[7´��^ �h8�X��3pT@ ���x��
ߟPhf���'�-���X���s�\r�u��O�� ��j#�Sà�!6d�����]oB�!d:�߃9������?jđ<��	2a���x���"��		i��YAc��pI��:�ĶP��&����+fnG&��+R�BD�!�~���E{�C�O >o��@l�pA��41���G\PO$��# ɢg<Hh%��GK�XQb��qW��K�'{�aZR�N�b��򹩝1���T�bڍ����Ԫ���P�:Q0M����i����w����Z�Ʀ���1>m����L%���o�.���e���ݺ( *�d���;�@6#۬�u�Δ9Sc��N�q�D_�V����`i��P���#d4��������ssḃf��fx���~>�;FJo�~^���{����` RN�6�Q��$��4��4�Q ܃��9�= ��T� �~�6�F���u��5!~Gs\�5�k(4��	!d0Vo"�ܱ�@`�f?�r������-SL`\�tJ����aD����
�сʣGDjX`�/A\��.�c�9���.A���:7�
��Py�	n���"ˋ�������"�!���Q�	�s�~����p���ͺw��[�B�0j���>�c�Zcxw%�K�g�f_�w�����$ɐ�}�g������K�A�b���JZOSx�C�d�����[v���~������5|�$1�ߟ�S4n��[e6�����-*��LN�t{��g��nϮ�����]�����I�M%bo�gqW�}s���wqW����~^����Jl,�Y�	�y�y���q+����a��Tߌ����Oi���]�w�t����_�E�,m45U'��w���Ty7��]�4��2�2%�V��##�?�m�����n�����~����=�����D]�m8�`z�{݌d�7��8�/Qi_d��`�%�2��}؛<S��i��H��V89mTVc�u��ʍv�������J]d|[h��I㺽���f��*��^5�S�/j<����枉�d�r�^9�k��b_�ֳ"1�FH��[9E��ߑ���u�g0�������~�/N:p�ˈ�KK˝��n��8���y��]*>�IɊ�:Q���������"���.�"07��k̡�����/
�n���h#�l��@鍀�.����~Z��y�8{du��?A����a�%O����,Q�������&��{h���K�D�Ôd?S	����Y�sÐF1�^�z�H����;�K�'�6C�>P�CMM�\P�� �B��RD,���^S����i�)�6ڥE�t<�=5�o<��#D� VH�Zp"��L	С��q�7B��Q!y����m,HlW���}(�%�<�X/�#�P�}�NCxƥ�G��5�Y���*}�5�|u��7��pյ)�I�[��r�)�گ�$:so��Gs(P�o�c:��!�	a:E	�:��=���i���L��韜�%�X7% �[�FĘFl�,�qv0��m���>�)�<�w�#��P���	���Q)���0�K�s��)�wzt��	(�#B��<M�����-�W�U~��Z2��*��W5xT=��S���\�mj���tu���gN��~ŬРԾqN�T�mW*jY�!u-��mS��QG�^2W���J��]e+���]X���}�ب������<m��'O뚻�'�� �\�],�y(�:$[��t����uV��J�ݲTy�-C���VH�q8�[~�N�a��K���:o������z�F��|Í�F�B�+k����i�ujy��q�pm�K�������=\���|�#��Ԡ%*g����R�ǭ�lpW�uO�Ŷ��^*�?̫w!�{��Y�3�Nk��/|@�kᬜ� M���������W\P`��uo�T����y�!�����wi����r��뢗?cʙ�]ɚ#J�?�=>1jp�e������By�	\�5'�%v�m��6����u��Yw>4k޺��%s�X�Hbl�A�{C&�8i._2#�5�p�����o>�oϫ���v���w���/��7i�Dyr������,,���?.�8,TNj�q}��m]��Ō8�7���0���/C��3�� �wM�?@�06+�WA��n����~��Y�v��nd����0ȍ	��Ei�8�0!�b�h�!r%+Qsg�½\�[X���@����q)!�%c6c=A�?ɵqh�,Z_ �-X�x�D�����P��N(�Vt��a��2e42���w�љ�t��_x��? ��_�՟��5�\�:�;��A�>�M� nMxA�	"���n�j�9o+=��؍+Y�!��\%����i�0��m����e�M%t)He#��&
	�f�r\�B _Q��d1��
-M�S��	뮽��zg'�r �`��U���f���L%I�o1�h�(�!�Bw�"��JJZHY˒&.#�9��80��%>��3<Я	���ciQ9��DP�`�z�M.��Pp!ϡx�K3[���Y�@�k��\�61Dv}n�g���6�}a4���]����9zؓ��ue�Y���i?��(�0ij��;��پu��u�%��nv�UwO>�L��2�롛�k�VQ��=f��S'z*^s]K�{0�ԛ��z��xJ�[��* ��K�Ǹ���|�����Z�ᡙ��3�V�{���ÆW�T��x�|p&`�����#Dm�e���^��kw��:�l��)Nƛ���o��&s�������÷_l�Y�>/�C)�tܥ���OO~_*nb��Z=���T�Uُ�^�*�>>��Og�&,?��GI	Y ��
܆ƀ�=�sW� 4c���R vSx&g�3)%)�`P�`�=L �6�@���i����R0��ed�� � ��\�
C�0����˗�g�`A4�Z�~�l��t,�y�rXB���y�ư?E�T׳"􀄓��D�[�XSҡ@�2���&�v�0�e���v���dO�]I�`I�R�DN��h����
��ǾM��T�jԟ���
U��Ic:���ά��P�n���fCI>R����=*�-Sʄ-5⊔;'lX���0|=�%��In�{.s	_���	5��'Ú�t"��O��nv�-�9?�K��'���b>��6b�����
���lf��3,��krx�|vuu���'ǽO�=9��0�t�v|M�ͬ�%n���2F���Y�j���;ï����̷�����7#eS�v�ݙ]x�[H	�^�5G�?`˗��~��O�9����]��w,�a��t����2��:V7�5+���pE|c�}M���
���Y�n�a��(��ɞ�>�6�c�H��BC/�U��N��ea�9��^��U�yÛ�d�խ�NF����t0u���|׉�~�%n�f��y�d��g�F�:�l��*�_|�nPVf��7��ݒ�$��=�\�J�î���jf��������'�g}�gU���1~��-~�|& �s�x��s{F�(�S**�ސ`�������֛���ʕZXû.1��g�����T�5��pI\�\����D����j2'}N���:nf�Lӵ[��\.m���g}�:���57���o GV�,�c�4�>�&��	�,�Hɘ�o$)2N���逷Ƞub\�ӥI����(�{�v1e k����If���<[X��̫J��Ja�Cl�ѫ�`��J��fU���?�3����`�ݰk�=P:[�����}k������GɁ� f��E�)��j�(��{pg<X
Y�F ��z�5ťs��ſBU �1�\b�]�����I�ER3���E�p�f��RAx�DO)X��4���D�q_\B���Vn-��k��Rt�׼Će��t4��s�A�ᕇ p�ײ���9Mz��=C�B����@��������+/�P�D ��w=1��05��՚���@b
 �0gj�� 5eA,�\z��
4�i�5��~��/���$��=�V�+-Ӎ+֝6�43�$�쪊��޾�Ü�O���%I�X��v^�t�fM>557�1S���H�p�Wks�IO��l�?�����N�	{0Q`�Q����=�w<����wTS�d�patk����Ɲ�W���k�c<6�V���;n8-;��;b���k6�Z>�����v���Ɏ�n>�vhQj�����,b��ymU�بΘ<gX�����=Y,�1�{~W�� �a.q�XXn*�_�ܽ���K���$5��L3�ݔ��>�B��P�u;)�/�z������i��H���	p@�"S�0��S�Ǵ��pA�T �TŌ��\�����<TQ2��$# �<q�TB�H�B	bqJ��bhL���	Ol�q����[�t��qN��9�t�}<(K�`8��K�9��� 
�b�m�B���_�#�����4��7"��z�� � ���t )T�[̇\��BN�n�!٧�J���6����a~C^��6b� ��γ|1���Q���W�J� ���x dPc-�����j�߰3ӊʷ}���'�uNdI�F��D��w��I�@��U0m����� x��m�c)�� ��JdSlj�U_}�)��5��y8�oOK��έ�s5�F�$���m���Ȇ^PفL��d��A�	鼗�CW� Y�����b��yXF5�^�35�)�	U��p+W��^�������	F��m�x/ �J�#h5JbD����c��UI#���9,jF������ ۣ�Rݎ�������?��{�0�(�ڻb��ۀ;��������[lE׬1o��tT��3�/޼k�G�E3�.Z�4�n�2���}|�#lL�w�G|O���pwȮ������P��%,�Zum�l��y&c����Z1�O2��=KE�������	yU�c3�CT��ڟ�=y��d�t]Z�"�ph��v��(���qS\����4E'�>[YzM�Ʒڝ*M�֚�N�8�d�����P}���㹥��K�Y��l~�#;����@Cצ���L\�0c��Ǐ�a�\M��t��W���9Z~�cY���������Q�@e�.���.�DflJ,Y�G��e}p`!�
�/@y�f.�؎$�O�I���f�/ �gRgOm@G湈&�6<�p�������J��mܑY=���y<e��@j��GyH���^�L�r=���!�]i��J�\|�{k�%����x�JR-� ��X^*�-���a{++oC�@�ژ�BBlJ��^�	U޷������ÿ�
Y��/��1��j�}��O�j�	筍?�a]��}n��\C��~{�=�N�������jhm�[����҅�Q!�[�����ؾ��%�m��Ϸy�w�H�/Z߷��������/={K��{��/VǗ�����`E��{Wޫy��~��� ��m���xd����L/?8�w�ڠʍs�pp=�b��w����3�}��ʝ4֨u��Q�y��)����B,�%��$=Ec��Ҟ��&!�"�D��w4N�w$�j�Y��PHS8n9 �[�g�/ɠ=��dJ���~�����09�{��brd|�c���pm�d|�o��k  6~\-��^�ɷ�Y�ɃZ�y柟�Lb��D��{{ڷ�^�Yh�|gmh�gO�/��[.�\�]���$�J�D��;w�|1
SVi}�VV���)�Zٸ���Wt�)�Ky��|�ʙ,�W��6}���ד�5������b���ψe7G���	>I�vk���7[�X�*�g諷��q�ᑈYV'��o���{���|�g����MU��[�
�:2�������{�H�)�>���fm����3]��	�Z�nrd�M�/�l�R��j�
�ԏ
�T�3U��aǺ�yOJJPR�K�.J8|n���-3���ͪ�hj��o��\�r�d��޴1��/W�$��z����W�@-�/���,@�!��}�swrٟ�9�7b�Ch9Q1�ԁ�1 +��z�_�����	z}A��*�B)�SL����J��sJƫ0T��T���^�aF��?�1���q:vi?0�9��n��b�U�fx�Q��0'"�8�R�E�0�!�3e�8��*��"GA��a�r4��Gܰ8�	�܂��G�Y,	��}U7�X�a/t�Y�/61���e3�S����+��gߚ,�
��G�*�O�m��m��f���S���Ʌ��9{u����pt��ZxA��I_�լ�8l�E�Ëڰ�`���N}� �#�����?�.C��l�d��.� bo��x��f�s-(��n��!����Z���'�;�a5�_���_�LT`*�8v��f�X���i�^���1�?���R8�JHa��!G¥�.�Ѝ*,i�Y�S�`��j��3�1�
�0E���[���H��L�W2=Qԃ�/�gP�?�2�ZA(k� ��Q�ntڛÜR� ` ��)�G=ۊ��28��u4���BέH ��?�n�A�&�~�`5܍L�B��u�%��Wr'�&�݆��?�1���H���6����l��;nђ@^�Ԋ�[�½Q�C���e_K%����) �Ǘ�xf�����Ro�H���q���>�=�f�SM`|�����b,Δ��Y(��.�Ѩ=^_��K�R��๣Wٚ̐�U��si9'�WkzN������0ߴ���YI�2Qn����f~�<G�ë�tУf��^G;{h�5'���SLο=�����>%.A�\����ZvY���
�17A<>Kh���ɵ���d嘎�����Y����}ح�O�x.T+�<-�-�֩��e�*�,�]�Y��r�L�֣#3d�EC�ļ	Ơ|��1v{7��M��,
�������Q&k�"u�R�9Z&/���0���o���nd�@���|�kn��}�?�L�%����[����Ў.RL���Y��.�0����.� �r���u��	��Cp�o؎͒�Q�Q�'��F�qv�MF�c@?�� R�~��)��ǭp	v&���,0��̋b�`��YnB0�X�JáC�UN!���7 ��A�㣅RP	?�W��u��	*��ʠ'���g#��}б�kr� �=A�t�Mw�Lfb�-�~,6g&��M��8�
���$�/H�e��־�P�.��_C}�k� �O��O ->���{�^M/� ����N��o����G��x�d.z������T�эX �%�����`�60ԕ��aT�n�q$�)a���EsIcW�5WE"y��HS4�1��F�@f�N8 �4�ͱ��?3W'Bt�)�<��*��|�W���琟 |$�/<z�Y�/Q� �ibX��&�<� S��$��� �A�ЋV�䝝L��_�]i�9D��,�WG���B���4� r��o��dl��G�I3z_O9B��&����
 ��F�v�ZgM�������w�8�0���j���-��U+��w�B�]�#��+��H_x�TeM�^�9�f��%#E�.z�iJLހ9��gp�V��A�Q�TO����aDz���o���g�O�蔉N�|Z��ʆ~ 6���@���O��z@�IW7c|v>-���]y��
@�S�B5��H�K=�	@�ò>�j.G�5r!d ��	?_�<X�F�k��n�2�@t�="0��T��w���7�"����I ĨOŢ�B�$xȭ뺙6:� ��x� ��j�����Z ���g�����Y���q	Ė3@�^y�5�Mo��d�s���B�������m|���i)�/�7s��e&Ŗ;'���JP����0%���x�I)��R�Eޖ;U�Z[�x�b�ླ�:�4~{�_{~u�і�;�DgZ<����d��źN;��h�nws=�lۃ['w_m�� ֞�ו	��Q<oW�TU�m�S/�,|�k{���f��W��JH֮��q�CP�S᠚=�]��-ti㖇{���O��.n�Xn��M�J�oM�d�r��w����ԉ��fz���ݮ�e��(�xV�ŷ��Lx�ʓCvG�Ņ��N�~�Lg�W�y帑m�����:gٔD�|�����w>��+/I��u3�sR�^����LV�{$��㨕�]*G?�N�z�ô��\�}�kvtEj��em겼6bu#����=h}nS�Ƹ�e��$Uh�XQW�$Rĥ����۞s}g菜#���b�5�i�F$/��RV;�Qk�ڝ՛Tyojc���`5Ǯ7�i6_�K���|J̚����[tl���B����I��G~\Õ1A38�Y����lOy��0�����땓��fC�'{���е�GS5� P���`k��K���ǟr�f�g�~R9�k�V`��̂2���F��w��#0=���ߖ5J0�6(�u`,M�� Q|�WҺ��~���Ĵ)-����`#C���Pӣl��j��=�'?Ĳ|�,eL��Yu�� x�=\,2)��.%����#D� VH�Zp"��L	С��q�7B��Q!y��`�u���A��T,K����O[��=!����؈����&�<�
i�߂�/E67�<���tPJA�X��}�(1b���z伂Y��A�#�*1� ��BuB`.V��ʊa���� ��*N��7 GnWDN>`@Iҏ�1��8�xJ�� i�!�7��~ڥ7��V"ހ�[e�]P`�)~�5+��4P�KX�_gPU&2�z&Ծ�?�B�
�Sw����H�uT`:�/�3ক�'����uRǻ��=7hs��駶���yv���_���k�L��}a���W/g�Q]�k���G{[Snpȑ3QM^�e�6w��M3�G��K�(�	�4�?Q��d��w��>�6��P˳�W.���m�Oq��_uPg��վ!!�=�'�oy��r���b�pou���_�n"�p�N�J���Bn�o��Ol|�veW�����ё;~|ٖ�@����F���+����O����;D���~YW���q���[2W��R�x������FZ�y��a��a.�B�d��\�)�ܠ��$�}uI�d�c��%�v+]o�JJ��y��fݤ��|�|e�V�E�#���8/)%T.l1s���2��5Z�-�Vy8	��j$��vT���Y��I���;n(��-~���ȏ��n�\8��S��i��m�n���9�eE�p����r\6���흧�+5���xI��a�;̤�;_O��$�=k��9�ݍ�u�6����H��~a���U�}���K��4��&Wf�[���	Y!��$t�!�K���+���k�j��@�Ya���2�t{$@���^�"��? u#�|H-��@nL�0-J�đ8�	1�;D��{(Y���8��u:��̸�tTo�ƾ��H�a=A�?ɵ: �#�=����RƓ$�!X�:�]�F��d2F�4Yy�&G#�3�؁�A��+������,k6m�Hx�С�y؝���C��~ ��A�c�v}��n����hsT�y�Ә��+Y8(�sq-u�7N����>���$<k�!��0��$.��E�PB������P=��$[J��Lτ�iG����ǄТ��!�a"�h?�U.��h���B�So�
%I�y�Ȣ��t�������n��,G�Z�4qa�	5ŁA�=�~+"FK�+GB�$l�Xc�)N0L=���"v�� �`?N��b�utD`�BVQ��Pzc��4$�8jl�j�X��i����_ͯ��V�������ta�������)�+[X���~mh�|9c�C6��������9�y�p|��4�Q1^�<��F�И,=���ŋ%�F.��������(?�#�2)]=@�yһ:�]��O3�b+��_�e��n/�.�mVl�O��?Kj�+�6��at�E̮:���ț/���SB�O�/j�x���q��h�S
,tX�s���&��9_�f.�!3<��f��ȧ�C�WX\��3(��Y��e{5��,Nr=�9��9������1[���!�<�nAenCc���9�P&�ӌ�w���39Üq�H)I�
����;�3- �fؼ�p���m���KAN6t)��A�L�S�@�0����>��� #��Q�����a	P̟������0��)r��ڵI8�E�	W!$@��B���C��e _M��a��A��1	Dɞ2�������B��8��ќ�T��ǾM��T�jԟȫ�
U��Ifa:���ά��H��fCI>R��Y���2�L�R#�H�sŜ|��p0�lG[ǲZ3O����%[e�|����/W����s�&!��=�Gx�.8�;Ӯ����V��u���.�(�� "78�V�ܧ�[9�ӻ�i�T^#��H�����������v��O�t���4������X��I�_�j��hnȆ"���'w�p��<�q����~^�1��@�x�L�yg��UTJ�r�shp�9���;�{S�)��u1?zGm�q�����(���Ö���Q7�.o����i���mq�T�<F	�=U������U��}ڥ����o��j*����ʕ���g�6�sp�,�dی�K�4�����{a�z�̩��C���ix�x����I��B~��֢Ϋ��O�?�ѹ2��Gz���h�a�(̠dr��ԃ��*)���W�S��<��-�4[�����v۷��q�}��G����M�3d2���g�̫9? i���c�~��i�~ ������n1��jݫ��1�dw��Z����i���;=��IT}����c�����9E��O��	P!sK�Y����}�0cS	X�����kJ�����A�"�{ n��x��b7�?ⲹ���AYv!�s���¾��_q[�>,U�C���]�Tso�:�9��h�q���&
:TN~����f���^���OP�V��|j��pTV���i7����E�����)�V
`$��	�%�� ��|�����F d�k�5ť��a� B�`:�b��av�~&�!�9���hAjFB�0�����,1[*� Ϛ��"w���2�H1�KH�?��46e���Rl� ��>�Ǵ#Y-Sr^$�~��f}`�Q"'!��Ɵ|�ц.A�.�q[7˿e|���Y�X�����7��p�W���RʾUX���Q	�� 5ei,�\zu�
4e�*�7���/,<IB@��r�h	7k.箩H��r�v�i�aBܘ����g^�h_ 3�ʶ-�/��o��^��Xʌ�����5q۟W�J]vO퀿�}��̣���;�^]k5��lϢ�K�̇��e��y�C���_���k?�ٹi������o���$'ts�h��%�e�G���wL��b�$ޗR�5��3�\�_��w���'Q�뉁��1���߶�J�b��Ǥl�}.�9%W���U�C���v���q֧P��z+���p��#�ó:'���l���e^f��~�����(tb\��"{NuB�h�4��q� ӣ{o�`$��(Ŝ!RO���Hq 4L� `HŲ�)�%<u-x�
�d�SIF@��!��i4%�J���Q!����>�h6~��l~S��tf:�^�Ȣ'J�l���gB�x��c�<�_���McD��ne���=r���]@:��;�������g������Ӹ�ݽ87ضP�U�:���(���C(2�v�a�IxIQ�C K�B�)Η{�t�X�) �Q�`�|��K���4{ձ3ӊJ��*+�Q�A�A�'�3x������.߾UVc�+���u�[AU٥�߆Y���1o4�֮=�N�ٱ�������������'o>ټ����?Ug�&0e�;ՙ�@K`"���Fj�-�'��k��>U�1Y�36_��"�7�|�O�2�=�Wv��U�C��f�!{,6�ߘ�v�#s��+	(|i�,a�Rn���G�o�xS��v>C_ �  ?CX��Jd+��WCz��ul���{�=��o�V�O�.�@ݟγ��m��V4��m?�o�GĉE/�6j�Ԡe˼��mUܣ>G�����ēz��?����49 �yã�5�.8�z��;o�-�����٬���r�Y����³�:�f�y�\�Z9-a��:4/�9��fhAɁ��5��<��z�����{]�~�9A{���#JߛINۼEp_W�϶�{�RM�T^��Z�:i�Z�����]g���'����{�Y��K�j;?[I=qV�:��7�Ht������y�0�<���g/+s�l
6y�{f�V��V��d�Ԙ�ƓF��aJ)��Cq�b�ܟ����RGq��d1[�+�d[����X�=wa-�=LL���Bd��$\y��G8����I��o�T�����#-Q���SF���+q�n�=��Zd���Q���G/{�Xt�����{�hj�����=����`⺡�Gn} �81~4pk�wN�֭<���h󎀅C�/L>8-�hJߎ#�u�vv� �+�1�/�L�걼�ne�	�*��/�������I~�=�2<stJ��O��kʧ�p�2n��ٓ��l��`|������WN�XN�*u3r��Q�Ƀ
\Wym���|���kS�̻?d�2�x�z��F�K��>P�6�|��x�B��˲��^։5��S8}�\%f�� ��;Պ�w�
b֌�)��:F�ތ	���k˿�a�����.'~.-��@T�����������IF��\B�U�|l��k��ۑ��o겒����������wo F�)���=��>_F)�)�;�g�d,�M[$u���b"ތ2��)��a��Yd$���L?�G8w��ٷfű_�]��	���;h�I٭@5S�)��o������_�n{?��0�P��)3+qR�ᶾ�'h�����<|�{=:_�\��rK��)Wn�K���[/��\o�s���&��J�w+�o���ؽ�Hno��E@���B���[��h���ֶ��|Ĕ�������Y._��_�)x��4m�5
��T]]xg��[���_��\:j{���7��Kt�j�~��f�U�(�������B��M
v�~���ݎ�o��L��PK�@�
�i�� HeS�W��vs\a�3()Ao㡼@}���Tئ>X�3Q�K`ܼc᎒�S��\g�t[U�+��þ��'Ӵ�I h?O�;V��Y紼���]m�Ks�o ��e�D�Q�H%X�N�3xu^O���V�Θ�b�_��<pW�dBhli*^��BX�������
3�'�13�y?�t��~`�%r�ݎ�#�>;���� �`	aNDq�����a:�C gJ�=��*��"GA��a�r4�t$@��p�	�����ԳX@7�K��D'i ݆���������y����NE�w F*������c�ڀxŅ��x��^�Z�- �@?�T�1���WL�����<2����\��5<����O���F�֣'�o��jx{�&���dGa5�Y���ʔB��
o������pg�V� �^?�<�`6؍r� ��n�8^|h�X���Z诀@&*0�h;�`�S,�Dɴq/XF������)F%��0��#�I�Q(A����T'�)T5Я׬�L�H�L$�V�����i�<Dr�E����� F$���� e�K�����Jk���=�9���A"�	ۓe�,�� S���%�"9��3�bJX'�?��m�S�
Ƞ2�i%��SV6�.�_�F)����ѳV��M����$�}�@\ʻb�Sb�(��`}v[v�,�p� ��l�p��y����K>��P��S�];g��юw���Gf8o
����N���>7�H��
3W=�z]�G�o��+��%9Xk�]?�кc�*����7�Q?�8)���ң�D��:�;Y��X���mWf���)=�m��]arJ{���=���Ի����N�w��2����Z�m_���2}C�mV�?\'�cy�`y���:E�uDܓ���[[���Z;t��N���~1K}u(]9�9��I3��H3�As����%&{���x8g�Va��P�{.��p�݁fB3�Zn��_ܸ��gY�����1i�G`��h"2�e��7�sRp�l��[ܑ�#��nR�;�l�e��"��������ǯ������D���U�r���dd3vQ��3��������Cp�o؎͒����?OXM�0��"ԛ�`��~�0,�AA��� NS<^	'�[��L�1�Y`��̋b�`��Yn��
$2e0�vZ,�niX����}�H�`?�����e`�8t�FlC�
l�|�� ��0���8�L SpM���kp��ч���1	ĺ˄r�w_b�EK�YF�����R�J荶9����x� Y�p+2z=�=�Y���T�cC>�����Y�`�Aug�=�Ae ��kY�����{l�� n�R�;���A��?�w`�+!#è �E�8��0�z,�EФ�3�5WE"�8F�HSt.�m�H�	$CQ�bz��� 3?N&��+R�ۂ�P;z �����M�' �7�n�aV�K����ZbG����Ư@� J�eA�蟡���S��	q�`"DO��aZR���6~����1����v�з��/_e���8$� ��M��ۗ\fҔ��q���)|:
���I��/�c����[G=��g�i���o�嚳^i��_e����j����f��"���GԑJ�96<_����s����ʋd-VYu.�g�طIj�����_���ŧ�{Y��i9���#����' �6҇����Z� \��H5a�=��uuu��LM�tQ �]L5
��#�n<TQ & ����>�9\d��o!wX=������Ԏ���U/^K�P��6x�L����,N~|��F�.��z�4 o��}�wD0�鄰�p)p�t`��§���T��_���:~�9}:.9��w�_�w�Q��`G��y��w��kPY�������Ή��&+8�y�9p�Z�|dl&�c�v�*{eg�T��h�
?0����k	�~�>y|��+#^�g?��|7�N����|_��Ƕ�
������zy<�K�].�>/Ŀ����F��e���]>�V��h���E�4�o����K:\���oY�E�ۺS׾��ٜfd����0����ݦ�v�
z"�o�i>�x�*sm�k��^�r�ksw}�x��&�)���j�E�o{k���s��o���@]��9�f���*���o�����/�������ط?֚ʛ<:�������%U��WY����]��������8X�(�zqԛ�kF�(�^�2�\���l���X�v�{���K�/��,6��}�;��Z�I�ۄU��')�ؘ�sW���7�W��N�U5��EV��Nb�JK�ޤ�	�[|7�����UP�p�m7��q��7�$kWǙ<:���쪦�����l"�����[��#.&l�׌�m��F��m����3�=�4�����ʒ,���{V魽S�#��m �M��R.�)U]�N�0?��G\7���M���h�5�5|G����ZDc���<s(�$���_ry]܋��%��(�@����w?O���i�W��A�
��z��=$��'Զ�1E@�� �}����>2�iSQQ�K)S��E��:窢�v��<X9�Hˏ`��
��J�/�� ǧ�2C�B�n v��v#8B1�b�ı�%�'B�ɔ0�{�Hq#TX��?1$�aA�EU9;%7���ܕ��16�� %����L_}B����<o���}i�=�jak��"���pS8B%6��U�*�m�`�_!��e�$�!�%E�m8Z	�����N��s�M�{�a��4�i`
U(��� u���� ���	%�D�i��p���0���D��d��=Ӑ�*�1�1� 8?/o�M>x<Q���$S����-��H��ꐕ�ނU�����]��EC �G�����N��w��6�L��Y�3��Z>T�\j����p�N[�]�w�s�ө�;v,�5粼ߙh�&e����v�3}�Ԇ��t���TM�0߈/����緉���/�o[�o�����`�
�%K��8ai6u���3?���6}��吮���=~���R?w���Wl�h8φ�U����/���m�پ��	�D'*ܚ��?
ܱ�`Q+�Z�-g8�^��nE��e�Vx,��'��|�Ά=����6U]}6�/�鸤���(i���^��HY��Yͬ���������{N��:h��C&�v-]K�R
�������UzA�Ϻ�k׵�h��[v~J���̽�U��*hX8ܷ"f�A����?�?�8 �޵"��|,W�	�*k����2�ۗ��F�(�����������S�7L���H.}������tζT�.=$��l��e��}nޠ���$���>�y�M��������q�{�8y~��3�u�)�ٔ�ښ^5n��q�~F���7}�3t�(�&��L����o?$��!�K�0�t�	��_��a:lV�/� �9�	����W�H��@��:�R�a8�F=L��*q$N;`BL���|C�JV��*���z�A�lX���f`|�0E�a��Q�'�"�'�6_j�#�X�x�D`y�;�>�:�Ru|��k:����2����<Q��fM�]�@FW���m�NL1���w. �H(��w~;n �@��9�G����j���	;�=����J�|ű>�2�i�NL�#ۜ��r�L�n`W�Ͽ���/C¯~	��P�D�7Y���6#x��$@�y��EBnP ���$�'��D�b�@��T��¾�8J����E#G�1�:x�\N3�]++!e-K����愚�� �q� g"x�_9�I¶*�S �@q�a�a�6��9b�@���~wv� �8���s+�.d��g��9����.�������x������sB^��M�G������xHF�y�pK^k�>�ϓ=�U#R�V��h��M�߿�����e�^DΖh�3W8o�J�g��+��x�]��C��WO�R�H}��n��yn�Z���%Z��<��RC�*%F��7����;t��_�J��M�Vp�Y�˼�����g�oO��4��;�qt��r����3o��?{�e�a�������k}�4/[Y�)(_��`d��$x���-y��KՂZS��NnrѾ}e���+m��2��(��/3�+uX~��Oee�� �6� 0���П3x%e(�1ݷ�2�)<�3���K�������?Q�`G�.8�aS|�Zͽ��W�KA�/t�������4H=�tX����� #��Q���9�%@1�t�P�1/���șjt�!]��$�M�� %B�
Ś�|41���/C\���V#J��qؕ��-Jd���l���?~Ѩ �~��D��QA5�F��Ё��`W��8�1�L���vؙ�c�. � u7�%5�HM��f˔2aK��"��	s�1�C����V?���֖�l~Xt�dD�L9��ªk R�m�ט�'�7ͼtß�a����2�BC�Eξ~�[�֪�Vu��r�����+$τ	��o��8�Ι��S�?OKK�">=`�Y�Ӧ���;h�Ŧ��Vɕu�$\���ܻ͟��e��kj�����T�]٪�I��C7ػD��J.71�� $Ƿ������u�_���;bxb�s��5?���Pm���u:l�����z�;[7m8��^ _�y�n��I;G��/x��0d���7�dd�L�Ӟ�99)������ߧ�)�k�U<���n�x�|���"o	:zo����Ptwʠڒ�C�J��N<O�����H�Q�歋v6���w95��dU�/_����1v���0!�fY�е��;ܹ�Si�1T���W�ز��~��~L��wע�i/�l�<��ѡe�KM����_��c���{/�~�Đ��Ww�67ݫ�Cuo�l�]��'Rf?�+���us��mԿh86��������N�k�4-W�*�n4��܋;d�`�7�#������ f��"KĄV�dQ2&Gcb���qc�t�[d�P鋗��v�1��1�1��3��~}���������F��gp)�)�4u'�sp����9X�Г/��n��1]8hF�D'�����!�;|
�x��J����#L#:�g�K�K�| ,)���*�r)iJj� ��Yc,��t�6�#7�@XS\,�G,�A\�_1����)ӊ��(��f$$c����0����
𬉞��(r�i �+�,�}q	i�GB[$�w���8� �[��V:������Ik���V2�(��)�S�d��LU��r��4�<�2A�_3
>��5��`�w����f����w��7�"� �a�XwH;C�cq��Ћ��M{�Mb��KO�>F@���f�}�X�=����mk}��V�7��_���ݝ���ɐ{]�\Ϟ��v�k�5~G�-S��d��O�/i��c����9)�����Ⱥ��L0ʬפ�����#3����g�ޑ�;D��r��S֖z��8����9e��������Ok��<�$`�R���.��w��-��%�
��-�ѵ��G��d���S��y�/S�*����O�X�����y�
�g��U��x#"�.FzE�F�x��|iʵ��q�g�94N�BM���t1�'��c\w�"3�:�P� ��-O�dZ�����+���ֻ��Ч��*�֡�VA����(��Qfe��	��<���0%!dd8���'��Z����W�V-�}{�=��M@�}���ܳ��׿�������s�ʧ��^l%��BM��cʯþ�Dn`Cc;��G���k�-<څmxt��2u�%�P�7D�yˤZ2��[��˥/��?,�����Iv�ې>��#�.|�b@3	
r�Η~O#
'�s���$��?�T�jr�o�����������h�5�a2������K�I�=NEw�������=kLP�1����;��KB7Z0$!��I��Y�~i~-|QF�;Dt'��k�8lދ皳*���Ϟí��q�.�\��Յ�ל�Ŕ�s/����Τ�v�<���O]X��5����{�|v���r��^w�p���?p����Wm��A��xݟ���ø��~�M��sw]�6� ���,B޴y��[��jv�g�Ňkp�ܜ�3ܨ����"����5b���'�?���p�s$���Y�O��XK"��Ӆ��%��a��r�?%�z�O���b~g�ɴi�E����~�|q�zT���Nh�'�g�:<h�⩝M��H���c��Eq����	��o������Չ�/�싣���ie�G���y�#�wꏞ�����K�Z�d�'��/�=�����>(+u����|ɨ/pÒs���;m����W��m�WUܼ�������UE�_P���0v󋏿uݫ�w]�70���S~Uum�����y�w#��_��ƕ5��]��C�F����:���?R���.�[򮅋������//����n���矎�t��	#�y[��;ώ�������&���w~�t΂}W>uN[h{��#��l�_~���/�յ����O�z�ӻ�]f��~_���w�H�oU������Yu����ȱ�PPOT"d�M�N҈������,��(����oo������q�9)�+��o$�>�'����R�~[ڻG#)��Efv���=މ3���<�ޖJ��������K^ɜ[�����wNN�9uȒK�@B��u�2�5��.|伮C�M)o=�5�O7�6Ս��vY{E�����9}�����]�߫������X۶�Y��9�t����ui�_F_[�%��-?}������iGX�+��7�����������:`�����7���Z�ҏ����#So���s��zJW�g]�D����?>�_�������-���O�\�S3����ǎ���T�u�ko��ws݇�'�f�����nz^���M{�76�I�8�:��Ų��3�D�cN��nig������t<��m?w>Kf���C^��(��e7�;!=��p���3��rn�����a���s��2����?��r�a�\�k�$���O���bmv��_����
����ɯM��>q)vhM�ߺ�0?�r���Ͼ�q��M����G.~>kJ��cg�{��>�Lx�X�'����IW��,����{�f�zv�w<�iP��?�<����棗.Z��mO����������t�xiU���a�}c��+W�?���]9-��X�ｫz<{�X�+�Խ���g��芷"��v �x~~��y��gL}eOKŢ��y`r�wޟ�ˋ�Yx��1������/�t[��!?��셻2�E]�n�'�x��ˮz�wެ�K��Wq)���K_{����o+�[�u�v��`�_�t�F���Q|wh�=�4<� ��fH��r>b�z��9�Y6�V-Ӹ���qp3���X���h����S�����̀a�?���A���бo-?q�e��؄����=��=����)`�,��ER2��c{z�1��f�����"O���t�o���݉�r�!���B��U��z'�kR� S�j�J$�j;N�]#�gY���K����t���5A��ӹ� ��%�]
S�����N���LEM�D<H�1 X��<v�i��5j<�֊rk<�/�����}���:�rH�u}�O�J�SƠ&!���tp�p����5�y�N}a*:�9ڴ�W~��� 7���}��9�>w�������i?��&����8���\���,���c��=�\�5
��z�[pE���q^7�P�����|1���l�sL+�7M6K�:n�ۈ8�*��*�Us/�=�;�ԉ<v4�]�����6��d�Y�Y�68Q�v�:_��g��)DR�[D�ʬ�~�*��-ʌQ>)�:\ʐ/5N�W�q�
��X|h����GA.�E_b6E�//�×�V��P�^��a��Gq.���_^�Uz �&�p�	���<s6
�ث���v��~P�Y�����c�.&A� �9��&̦�0V�㭇���ǵG��o��^�m�հ(Z�]DD��7c��QmZ�Kϩ2 R<~����k� |�4�nջ�y��{ ��k+�_s6}����o��J:f��\c�eu�&��~ސs���G/�޵�{Oٗ0���F�̉��Y����O�>(�ʼ�m�*�=�k���zU��k��<�䥳:�#
�����cj?�~��ǣW��u�}���͙:y@Sޟί�#kHye�����]�g틿}~@��'���4����|��;�l؞�ļW��袯�]����_�f��c���wq܁�.��x�Д#Oο���I{�R������]]�xǫ<r]�Gˣg=r��ڝ_���������ڵ'ǎ�p�ݧ���m��+o�������6�G���G�s>w|8��^��U��o��Z=�#�t�s���7xč��/=a|��u�����>�+����g�`���=�5���	u�fR6 nF�@�'�$��_2^���7j(�[nqW�b$�a?�|k+�\J?袁k~��k`���bѪ#���p�À��.���>��`�Uqv��Yu�Qކ�svt��6Ǹ�U��x�zd '�T	B
�%��������he�$%k=��+�Ǭ8�O�y���^����u��Ƈ��Iq8!���IW��<)��,!�����Bǭ�
9��wn������;�,�IB#��`\���(�(x}���C8�may:�3�ʯ���;z!�1�0p��#}���SIьyl	�pL�B3�����i��Pɩ2#����36�G�k��a�a��� ���^7Y�1��]`�>��N�D ���^*��J�0F�̍�K\�1�x��O ����#s���hX�� ��<0a�X��Lˬ(J:鋝Lq\3X��|��"��wY2��Lpܫ<n�J�߾�f˸d�U���k�����Һ}a�q�*���O�R۴��eO����a�3��V�f�omnks��	�UVTn�����ʖO�DD,\FHdZYUYAvRRf�$��*������*/+/+ʫ,�df�q�)I��Yy�Y9�1)���E�y%��+�23c2�r�}i�)�y��q9I�<�kh���mH�/Ȗ[��T�@��$��˗[�d�5�	�yB6t�|s�kǐ�_���1L���>o�$-�w����m0�3"�}Y�de��/"�L��I?��f�N Q�x1����[�k��ܤ��X���]��G#�'�� �������<��Yξ�W)�?y�`(�֓�0� 6&6>!)1.&6%55-%!6.�6.>5-llbJrJR|ll|R�x_ZZ:������Ц��Kq��uܯ��&�%��|i��QuJrBllBbBrr<�̟���K��n[�'.I������0ގ����������.(**.���)���� 7;;���0?'++��;R�STR\��bf���J����p�|v^!��gqR'⨞�⒢��\�󲳩�����-�6��rrKJKKі����X��v�_\�㹾��������M5��4��cbeΦ�;jk��T��d�+S�\U����q���t����_��XfY�0��M"���������1��9���6ɰX��CH/�`-e����~�Tp�Mf-�mSd��3�b/�q��y�&2rQ$�'��[�/`��GF����9������r	��ה�!FB!�'s}��߅oN�?��Td��WP�u^�^��A�Ve�	=<d�0��uF�EwXG����f�9aӆ���u�qԢ�Y�9 nV_?�|><�G>�z����w��~��;�|l�24��r<���>*�-��\��VB쩝#�k�8=����V���zB�V�zb�%YN��'b>��i} ���/3��4h)�ɚ-ravࣛ�����6)p-�B;���bXE�)�
zP8�`��Ds���
����.y|�C���X�7�<�L�E�S�l̟�cC,sq+���j�96�������;�x�ڟ=x�г]������C�i��{��gt���y� ��C�>���y��z]����|����!�`�<T}�<�rWG�a�{�?��!���æ}��(��zރ�*�!�e�e��Ĥ���z��-Օ�5��T�W�l��rz�R�����rӦjj+7S����Z�ی��JJo���U5Vש�-,��m�x毠�h�Ζ����ڲu+��U�ڭ�n�햗�(h�L����,^�7W�+��+���?��O-���۪��h(���� �L�����!�o� �kB�fq�x`aj����	��ٰ��F�؉c�Xp���7r��V%}��_�X�Pj����G��;Ͼ��~���]|$H#Og�$V<������"�,����A��A���u�xGB���G�)1����	yDSWM���P
��dz2�xVIAHH�{l<e>/*�4E�C�D�����Y���T���&l�bmm�j�t �'�ƙ�'�&eG,�BVۚ�d����|�'����˂mt�n8���.����"��6rJ�f(�9��W���1ҜL������96�`��)H}������P�F��#�����oE�ȑ�q[��)8����18P~�����D�XJ�PpM2��& iZ��b,M�G�RB�9�$a�^ �)�+H
y^��@q�-���L��VT���L�v��c�����]����C`��qj��}�����q(;�Dǁ�y�cǷǍ4N�x�����q�v���S��Ǚ�q�������:�=���+����`�����@m���4Vh��׍���MX�Zň?�F�)dј��>hP�B�-:]�������K�)���#Y� oq�`Iv�B^��0�'��!�_�����+��,[��*�2S�8����/�@�������G1H1�k������(��򹲄�"����<���Ò�:��FD������n���)�,Z%xr�)��4�$l_ v`b?��q����Qw�Vt�?�3?�>#t�~̅{�?���}�K�e���Z`��ڬW6٢�s�q��i����%e�������fsueYE�f�8� �2z�_���������ӽ#7��	`���S�q���` ��D1~Q!�ob�e0�;�q/P�l��ڭ|���䭱�Eh��m��;w�m�mjnni�Q[�,lK�ڝMMM��۷�76R�|k3���m������l����Ό��PEC]]#�i���j�����omm�Y���u����=����_�S������j��}��[�>(1�%wj+��%�a�JՍ>ȎI6h-,�����ONI��=�7�ۃĄD$f��g��egf�E������OWm�^�ny::5++N����C��U3_5oc���vԙbN��?�Jb���KV-�]��շ�'HFF�q�;�/�/�Mr��g��*�P�o��t�:O�+���[L�j-$#���u�fb�G�zx
k+eJ8o4���&i+�Q�W$\��
��P6�\��b�d�
^�^�	ѲHr����u�Ek�n��N�F��qQю�j��Ҳ�rz\�i䙧l7ãs�,��OL2.9�Aȼ�%�3q�~�}Q( ��=Y�f�_�^3z4�(�H� ���|8K������ױa��@8X�n�� f'y,{�4"�$d��������$5ŗ���������������ҟL_zN.�pe�|����Y驩�YY`�s���r2|�ԟ���!-�hy\Nv���!�`�<iT����匴�<�g�����f�S�g�|��Q��l�Ϋ��hj��?§��-Ѓ4/=㛷p�d[ *�F�.��a�&wa���xQ��m�����[�c���]2����e�nK&1T�D���g�TKfr6j��^1�/�q"='st_�"�G��Ʀ�f��+�ऐ(�@�f5�FH5��ɡbt�i��I�`~�>�;r>�)<\�S)q�N��0^Ó:��)u���X���L>�A�p�O�#hat!:�$����g��#��@з����"#c�����q��*yyUdsM}?�I�w�����l���XHb��k�*7�ƒ�����ݻ��ta�ʈ�͕�e��0��AVE9l�v4�l��M�[뛛��2 ��w�0�?���S����25����N�0��5Xl
��M�o �7�����>�M�C6��H���K��8�UaVH�^UX����Y�f���Zu2k{�� Ʈ�Y*s�8�]�`(��{"O�GC�(̐0���)t1f0	��!#�ǽ ��0���<���C��q		II�q)|e*�����0o2%%59>66!)9%9�c�g�y�il�d�j%ƥ���d't0��ϻ��_����<Jz������������<��322Ӓ�f��MKJHLϤul���KP�KO�	}����}�n?�?R<�ύ�yo|����Ç���K 4�L�y=�#P>A�>5�<�'MO�?�����f?͗Ǿ�d�(Z�5B�Iž���T������H
�y��0(o�w�Cu=O����j��<��D���'&���&����a}���Mrb\L]��ƥ�|i)�0om���c���hSy\��Kbq\gǣ�yɴ]�MLJ��ı�		l^1�c�q�1�g������f�^<�SͫV�j�tjyp�л���s:��e�Y��;� ��k%D,5o��8�2&*j	��������͖�%k��fED؊U�p�_� ]g�����Fgdg�ٸ&#�0?=�}LX`���I�;��Zv�ٺUަ�rGÎz����fG-n�Sw�4�>[�l���Q�}�Ʀ��ƺm��n�mln������Ɔڭ[kw4P�㏨C[��������*X�߶C�5�8�3�������v6�8綺�zj�T_[��9�k���f{ͼ_"��]��`�5��� �s���d�R�?�G�X�D�Ƅ�0��+��:%=�Jx&:)!9--�AWDԽ�l�̎N��NIK�[��g����<|!m>"��XS�C���y3'�!ç-���\��a6,�F����18�|���3~&�b?a�/�^#����T4�<B���u����0Fϲܤv_b�E����"p����}-��`P�:���v�_|f�(j��'�A������N4X�쮲i�(��3��r|\��p�Xx���C�p����>�w(%hǌi^��ѷ���s�e�y�G5���m�����{c�[v��7�ٳ���u��(ʪ,����w4�jnV�cH�>����-C��	v��y�ͦ 0��.bn8#��B���t�M�B�Z�a=�4c��SX5��aR&b~8��B��y#�dhj:�Y�?�;:��v%��	�~ӆ���h�l'���Y|��i
�T�Q��2��_�ʯq��2c�OJ��2$D��4�d���S��z`�9�)g��Í��l��!�&��G���;�7] �����j�C�6a�&�=�3g!�y��\��r+��yB�2.��t\���¸U�\		��#�]i����V+��Z��q�����ji��yO����	Tei3zg[�AY��+���A����G�#f������0���݊�����Uq�����e��=�ͭ���;��nݵ_؎�-{���׶��eW[�]����_؎�vf;%q�{��3�Ύo��=��1Ϟ�ֶ�m���vw����=�vwtvu1�iڮN��ڦ������vYNr*s+��p_Wvc���}m{i���!d�{*)-Ŝ���v���t�PU�<���_�0��li��oP�eSe�3���ل�g�d������g �&8<��g�Z��^�`UO$��<��1���g��]ioͨH�u�X���%�Q�G;�\��RQ� �P	�[ǩ`��u�������4y��w���X<�Ɖxf5���Rf�lV ��l�Ck���B�d��:�~�V�㜭��'���X�������	Yfo]�X?Ɔ�^��*���+r+)3�f%���:�]J"<���er�	�1<�ʩt�Z��n�i$#4�7P�"؏�X���U����^e�Pt�XT7�៺�AAB���7k�X3�m�n�B���:ˡ�NwP@�&��VtA�����60γ��t����ƣ[tDA�u��4?H��0~`�MK�Nku�(=X�n9��09���^/�n�[Q����o'��A��c�9 (d��9-���[����K[	�ğ)dK[��g��ĝ����Ώ��.��0�	���dF����{� �%��7�	��z��:�H�T� �+��0��u�_�uF�Ew����Q���Q}�ie���e��<�gƂ�%Қف@R��8��Мn��
��Z���8-�%��ȅ */�߇M�/�{<���5�����"D�pA$Td>��KP�]x��m������MV��³�ӂǖ���^ݠ�]t�x�pȢ�KH
zf�� ����&X]���	�ZpXoϨ�w*�Ƃ��e�`�tbFVъ��P�å�2Wi`�B�Fra���pѸL"����N�!/��f9<Y��(FB��K��]�i/�eK(B0,{@�[�v�w�l?�Heˀ��(���:l�Bw*�W���-@8��z -�%(1��=�g�5^+	�Pu��T��'�� -�D! �{�o³	@��4��`ɻaF�Y����oÖi���؟" M�^J��4XP��E�aj#�5��N��fM��� ������G��$��0B\�
�ʂǖ�_��gQ%�9�����()tXc�b��м��E��t�o��P��`����Aj-@;,��� ���
��b<�@�W�<9�1ֳ� 1ƌ��`�Ȭk��	��3��ur��NO��Uv��6kÕ���ah���l�-�A$��%����:����`�i<Xu����RiuUt�Lz �hN�
�C@�FZ�#�o<�Hٓ亾�`yy� ��+tʣ�K�1JA$<Z�G�Mj���0/���+j�Xd�|!x��[�k��R=�w���H��u3�n��·`�8�U���u����p�㜬}ԻB�g
����h�p�����Kq2s7�����$'�TB%��u��	���p�$��y�~���eP�r4�[��к��E@[��h^���:�������5�%39+��5�Q`� ��!gQĪ����_ب(�YO�an�CH�����%`;��w���)�x?���2TJ$���P�.P�U:�ߝȂ��m?@yy��X���%r�4y�T~���<'[��5���h���W"`TݿE.�	���y6��?Ԯ�~.�W�%*"+7��c~�i�^2$޷� ���٬W�g(�����I
p.%��)�_ܣ��3+�Å���<��R��d�NbQ�Q�C�d-����g��RU�9dL�P6!�8D�:���fJ� A\A!3w���u���@��}����n�@���q� 2ލ�B�e&�X�����pzTx s2�J�!X)���p�Rv����?�_�"HG�~����%���PaM0�&�,�S��`IJ7��<��&��k�l"��*���3k�0��\�EQ�A:��xi�>S��*� ��v�{����z�K�]�lB�ӗf��T)��B�k8�|�ބ��!dT�@�t�����Y��i�gJ ,��;�ƪR�>�%�C2#
��D?�dK,�,���xӅ��l'��%�R
�S�-��SeVc�H��$e�(7�c�dE!!8�a��ݍ���V ���-�C�;a�=F~��x������d��)�'�
�D�^�𴢭7�ل�� �<�ъXf=�'\�����Xf�X4D&a� �·>7V�9�x�;�sr��Pt�kA�ū�$�?V3TREE  ����������������b                                                  �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^E�ah[U���7\67K�U&�[��Ȭ5��z����Q�N����)#��ص��2�[)��B3��2�	n�+UF�n�H�m�j�Ť���-�1���<���;���s��$��y��{���K�H�xi_ݵ4��Ȱ���$evuJ�Hs4�y����|XC��4b~1�����4�~Fh��-���LcHG�#���9<��R���:��L�U]H�e��E�`���/b� �Q�t���y�%L#8���F�� �-�P���G��	�D�Y#��5v`��*���&��A ,ts
O1�(|.�Co�ys#ސ��t`Gt�so��)Il	&{7'}��"V����%�r����<>�AP9��!��;yP����T�|x������V_8�4	�?�4�6<�pPiQV�v��~��pe����:�GW�ᇜl����n"��]�=Ȟ[��*Vu׉)��AйUC�RV/r ��H׊������V��p��$�����<�1xA;�=���v���������QU&Ü^zJ�~#��Õ���6�tZA}���~&�5��
*�u	Q���Z�tT�a����8�V�rԚM�]��,�4태zV�-��t��H�Lו����	�tg��'�u�J��x���-���K�cٽ)S�#n|��Gtn:��{�v���q��8�'���>3v�q&�I�e��n_�K���6}��7x���m�I�3��|l��1,er�Lsql�O�O:zWN��%�y���%�5�?(��m���3q��y�v��1�=y�e�h��$��e�f¨\[�	9(?��e��$5����ɵ7�
�k}azI�i=:xu�5�?�e
�g���� v2v�TREE  �����������������                       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           TREE  �����������������f                              �*                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDRu8                f       �              f       �           ?      @ 4 4�      lx     X                   �8             shuffle            deflate            \�        f   �           ������������������������T        _Netcdf4Coordinates                                              P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               p       DIMENSION_LIST                                    ��     �      ��     �      ��     �   ��uOHDRM(                j              j           ?      @ 4 4�      �x     X                   �8             shuffle            deflate            �        j           ������������������������P        _Netcdf4Coordinates                                        
   P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   ��OHDRM(                �              �           ?      @ 4 4�      ��     X                   �8             shuffle            deflate            ^         �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   h��NOCHK    B�     �       P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               6�j}           x^��	 U���w��L	�d
�BI"���T$d�2�H�$!�P�I��d(s*���%"!�Z{�s��{����s�Y�^{Z�?���b� ��q�%v4'��Ш��	�w�NY�r�>����U�������C��,�uN嘣c_���0C,��D����JL�����F�H�.�R�m[;V0�J�P����Um7���x��7O@��i�$4E+/݃�4,:���[>�^�"�&kt�nL�z��|�Vd�����4��ԍ"�`�Oc �G+i@x�u���|���������V�L�"�ʫ�����<í^��^:��o�tC��wNZz}K!�?�(��Ƕ�V����WB�1��r{��*U��w���H>N��lר'�]�+�W�G�v�[��<�/�#3����,���\jڵ��@]Y
 �۳^�EZ�Y�E�D�	�h}Ck���Ȭ��c��mZ�#��6�+����,�+{�����\-"2���ـV?����v��d�#B�7AjB��ȭ[�6��Q�����a��E�u��Jbb��f;,1��t@������)���;�,�|:Rc�k�>��(NN��� �i��'��w��mM���d��"43.\��p���H�M���#�z�?޾���ٷ��~��=p�@��z����Θ���>��O��Ō>666s???I�dj�����*}l�_?�;����D��]�k�����g��p���nI+[��]\rB�ƈ�5T�����_m{7N4ݧ~q�|g//-`tt�%��;I��65�4<ǮJ�I�X���{�l��_Pe>�Y�y&��2��v���pS�*G\wo(����ǋ�ؤ�ΦǑ��h&ܾ���_X(�p8v��q�,�y�'�u�w�?��x����U)))��4��&|�x��홢y�ŧ���z=���<�ݻg+㢸��e�ȿd���LHH���CS����?�<#G������y�2hmQ�_����IL(�E(ma������I;��W��ܷo�6&4+��O��7���=:ck�������R��Omo"==ġ��>g �8e�Oؿ����X�/^�OE�Ǒ�=�UT�����1��**S����n�T"p�dQ@�j����pR!�k�4�r�ZU�v�R`G>z��=8Ζ@A��j*�6*;ģ��Cjt�����?�(��L����f��?�d��yx��øJ����A�o���?�|�Z�.�-�E<��q}+�%|����ٯ(�,������ٔ�K/��ʦ�#�>}23c��m^!��2F��a�W������]�To<IAq�G�#���q})�MSw�^2ÓM=��"m�K�˖���.���Q��b�h}|+���`��S��M6^Q��N��7V�72-ߦ<69�/(����gx���?(
MmT��Jȱm��e��،�OR�8��FIM���/���<���rrr'��dF�'�^�%8�Ġ�`	��n�]޷��i0�J���|�p����h��4�?~����ngj�"^�,�eº��L�<q�>�"�KSeS	��d	1*�ˎ��CT����^����8Qo�J��ڑ�v�
��	���T��n�}�g����tt��~<����E��n���|�===� [�V��>w��a	U�^-�����q�K����w[�w�M����U�)�8�t珙�l��v���@�܅xm�����H�N�GB"4#�����#�͛7�*g]�F?k��q�Z���RZ��l��.�i0�+x�]n�`l�f�=|���O킝A�v�~�)����Q��8v�!��M$$��d�v�8�vW%H{��O]w�o�3�_1&��q�:�)};��D�CO�}>�8�;��e"�&�(�wr6��ą�2��'L��B�=���N;��rr
û�c�IabKd��l[Jmm}Տ�7`�FA�tR٣��U%����D�ZXX�*!H@��� ���6�,@g_����@T
���m��bt�̣���@�t�?�_������O����O���#��}8�}�x:�N��	6:d�C^>�u�����]���5W����;D������GOf���s;Lbv�^=���g=���W�b����VT\�X{�`y��5|�y߸o:y�9IL��I8�:R�F�?=�f�c��z"v@h�N�����ru��+;;[�e��_���b��~[+��}]��Ș�����>�t�q����/0&h�\"#|_gf;>��rDs�Ew�2�m��#t��v�y8�|hT(�M��_������Zm�֟ ��"�;�bVy͙�&�;�p��ϼ/p�\?�F�W���8}�#G�����Kh=v��*)�h���ܺ�{�ԁ�V���X��z�"��XтM��bY�o###P�ih�y��t�S	ؐ~�8���Ѱs�N<���88�����X]@�D�Ǥ��}�4ۑ������1��1y�����s:	�-*�ق��Z��Q��'_�h,,ܱJ^{�=��zxҀ��M7��J�Q%Cv��'|�A�~34"�nJ�;��C����/��ͥN����08���Yu�g��o�^���M�� [�-�S����>l5L��1�wOֽa��_�ؾ<<��(��U��]o�<yE���q���,x�bzvʫ������J��+�˕�BN����P�W�(`c���<��Ħ�h(���i� ,_P��a7.�����TD��1��)ؤ�:}z���Uv�z� 66�������)����-�mכ��h�{\���M����i�Xgu��)�1;x����z=t82�U,{lL>�����{�����`mӖY�:�4��Y�����:�離�ipB�YB���`a�\�e^��$�<������Q��&�-��<����S^��@�sV�]:������q��(щ��1g����^�%R�߉�U<�S� �{:�2Ģ�Do���9~�p@��������q�t�	�
���WRJY.�O�n��t��,�zo/Mw��c�⻻�>���Z�C2�h���{H���Y{^�t3VT�vm�>���R[j9	����QSKO�~��u�"{��?��Un}Ō�N�/h�l�	��<C�g>�����*B�� ���N㨈�v4�J�Q�Qj�Hz�A8Ǚ?�/�K�Kq�u�=^�)�P
&ҥ���-9��ƯZWWV(��va�����6�'_P0˟���u��:[G�o�埪�{�i���nP[`{}��2j2X�^�t�����]��������~%�}\�}J�6phZ��OƆg�����/Y���p-��f_���]CC�Ǐy233Ӗ��-�;��S�eƮ4�Dcheˊ���|k*���՛���F�˭�,����p��ΐG�p�ܹ3��nj����xM���4��9�M���,OOώ����"G��kA�:s����������5�usO3��He �Ĉ��7j9���s���l/n�`sj����Vl��ɞ�}{�
�Ԍ�.U��07�\T��.�i"J�������~z�;c�"B���ѫt%T7�d�_��R$�K�:XP�aP�g���TY����}ّ1;�`�-��{�����6~�E����+��� 3}w�>x�馠V�oL�iiA����A�n>g��G�w���D�f^��)TJ΢�8ա�ￇ�����v��C��(�϶��tK��Ν� Y=�8���r�e�u�������'����e��"�5�nxJ|�w��en��
Ã��DD���PaDy��D�-��F���)x�ga�'	�u��۵�+���u�/<|f <A�Y���3eP�^�>��6_���8�Gh� ��A�N�����d���"y�6A8<)(��[�F�qe2����%&��ڬ��Ǭ��X��ЩxSP�n���s������{2rG�>�����}��mz�Y4hz��O1��i�*�=��K������ֹ��1(FF�h��7�)�qޢ\(|�z��������c����o���(�FQ�X�o��\�wҮ������y...�� �%;%��Xi?����Q!o9�Jt*�oV䘟u#~eQbg�F��W�B�'��~*)9�ӿ�F�##�e��~��fdo�F�����R]�R���Y���J���v-VINH�UBO F%%�B��?WU�{���݅�m�6R�MwK��O�ʐ!�c��p}S�����������ҧ��9룰�p��ر���x�c�@���
�s�ʾt��Nŷ4�Ԡ3��k��1�@*9�{'��M�ܸW���ypdd�	EY^^T�`Lxy�\�g���ơٰ(!ibr��N M���>u��\*���؂ O>,�1��)0�W	�ޜ2�)��;�������})�޿��\����,vU�m.�]xg�)�B�)������znߏ�V��wy��So$"r�wj��S��
,��!��[���;�_I�φ�����1�쓜��� �� �°��:�AcyxG�M�?�-�����W4K��#�SMt���Z�dp>Ȑ��e�aoo/�Kdb�s�2J���ݻH����))y���q������i�؅�9Zu���x��Ą�?�d��"���J*��Ѐ�u���ˌ��ٽy���::������"��#-t$�5%���a�b�#j��U<an���Ю���E��Ѭa������?��ͥ��VM�?�.O��tE��0��w������ ��������\�65G�0ە.�q<�~������9��	�-f�!5:����|�Fֵ���&�_q6vޒkDE?�z��Nr���&d�3U��"Fa5Y��ʮY��Q�MK1=��?hϞ6���Q�6��Fy��s��V�w���cf�y�"hsI��d�h������uef���Ę�$,lܢl�d\(ejHK�𭨉�l�؁���&A_;��r�@p@pT�Bܔ:hO�500���O��]�]��]��$�_/rCΌ�R��PB���H���pD����#��h�������n���n������ gV��o�v���aW�8!����������s��1�
�0��U%�]3�7k���l�N÷K�߫��G^�I�F��nupO�GR����Z['Y��˧eeW��AS�J���,�,�� ���9D�T��V3�Aݩ��3|Y�e��g�[�˛��
��̱��˗�P�����2�MQee������,;�Fp���o� ́LG��K�*,!�X�������2AA���ޖ���)�b�p1}���("_�f��	����r�Of,�U�>�L_�͛77�::DEE�����`�wχH����屢tI����6�RK ���mD]O<aX�j��D��ŏ�;�?b�{ǯo��E��m�..i���DV�4�0�a�B���}Y�W���_�W-�3�g^��3{6��U��²�_;��|.���>� ʠa�%�%�+M
�w�wX��O����l��i�ĠQ
[��&ttt|�jw;�=�j��p�1�N+(����+Ϻ���1��q���~�M� ƒ*�Pe�O���}�
+�|�n���ھ��x�1\!XM�O..���3H���G��H�Դ�~硪UUU�߫���d�Nt����!=����ZU5�v��i��!�,� �� �� �y�_쿿^9�~���$�R�*o����EW^0��cX�-�;�>��.ou~2��׷z�3s0�90�rG�dѾj�)�m��B�\�ph����f)<��1�*�Ewo[�ӣ���f0�Y]�ġC�h����{~f��g�sߺ1mr6����zi���y�����!<C������:�1����yW�0@���[��Y�W�&n�ZN`Xm���&�]Ĩ��S�;Vׂ�M����5n�{����������q�ZsW��zK�"=+|4Y��wb����S?#���`��R�/'������z����=��25T���uy߮}����<n��E��'��!�����.�w�ջ8P=0�Y��+��^��Ӏ�3L�g�H��PQ��Fi��D``KK�FC2X3Ҥ	Cyn	��/Jgo#�N�oh�K	�
��[�n��i�~��ۻ�!&���� ���R�(����89��˃�O��
B��6�����>�p�ߥ���Л]�*�dT�fh�1���-�����$�l��.��"���9Q5����PGm��Dy�/� Q�R�f�������A��+���jw�m�+&�~�(�+I�=�2��RU�r����+%8F�J��%�?<:���:�ޝl�s���.�I2���+J╅�'jk��3�ǵ�ՠ|�ߦ�E$�u���g��Ȭ3�U�~Q�նV�.��������#!��3D[� �a�ɹ��	#Y$�ʜ@�
�����oHӾj�bI�������ʹ�!������Q{�:�Ь�-��4%�aH�c�)K��x�<|;�=�9���+�n�F3�/��Ĵ0�m�Q�={��0"4z��'�n4`�����ettBU���^������	��hA��G��1::7:�u��(�c�U�w�J��X��{���)8�a�M1��� ������3������s�T�����,|��x�H)+�n�`����,�n@�s�z�={$�����V�[��z��Lx��ܮx{^�s�5�q��5#���v�G_��r��c�]��>�C�-�F�6��n����T�ط�e�,Җ���H��=ݨAjL��F�)����2 ?�u�xҥT�:�mzW�8�GB��Ya����
�A*��0�܂`.y�YSSz%?e塡����yy@!�ͤJ��D����}�&��چ�G�a�b���>.�Eo�Uh���SK/NuI}�1T����[���n<9���l�X�_x�e�*M���6S�V�`3@g��%�����i+W�<b
&K �?s��½pV~nN\ܩ�pN<��		t�j#A��/�������r0B+�%�<��n/�^��III],]�͊yN��zTT��^gfc��͕���:L~&r�����W�Hjv6�&6�g!�Pv��(�%�X�l�=9���l���n����S̾Q�p#�T@�݁��-��6 	+���m2�G��
)����6	�l��}�$[FO��o�Y��3�ɑ�=� ���2|Ħ�,bc�Ks�ֆ�Qm�"��%��~���kV�ӥ��23�.�Ak�!��8�--8�&���X,t��7�����'%U�[���R����9;1�+�g�0z��U��8��vPb!=!JQV|8"�`���p��M	q��*�-����N�������췿o�y]>~F�u��S:�����
�����'�b���B�[��NOW9ut8��18�ȫ�������w0с� ���?>��WH��dkz��� t�?\I.IP�N�ɯ��Rn��WV�V�h��� ������U���󘈜	��	���1�������� 1dxQ�|�:E�&���$�or�6����sy!�u����x*!	]���?���t^���n�j��u��F��_'��1�����y�%MF�չ)�[)�tX�ת1w�
|/b&�-�ɶ�h�<7F:/��0<�aRK�[�e�;{~0�F>'��$�D�ю���*���Fh���'�ʷ.8Ŕ��ܫ[�{\IE�1��l��?d��
Ea��-$�I�l,Z
F��J~`�(����9M�(lqt��~5��D���l�yr �Έh��M j�v��y �+�-3s� ̴o�D��WW▃�[��X�&�gd���[�Q���[�l�q-7���?SD�2��m��zxS[[Gw��ƍ�d��%Y}Z�r��A���MHL�z��_P����s�1�Y-t���u��=�2��[��˗����WC_�@�'�#F�G�����o9�"t�(=J-�������ŕ�tM�77f����c��$�d���ޡs�7o��eظ9;��;p����n�j/17_S[kjz�H��b���1���B��lc��|~q�PUէ!�H����������C�y�;}�*΅���bgW��A��.�(]��p ޞ��e$oCF�&���n���h&�4S.���h���9�5y\�J�b�UPXuDN�����;===����Y\�+'�}�Oߥ�B�獆:�'��u�y 9ۻ����S]\\��T�����'尽���l�3�.zsrz{��-C�EX�f)���F���{ z�p`I#� �c]�K��AuE0&-����^���	A�'���zo�5����zS�z_E�Iz��<8�x�7$4���x,%���>3B�u���h��������N�3���5�T"7@�m������:;�^�-�������r5ۇullh�+�]�9����B�q�	 i��ۥ%��ݻ׀�����K�Rg�C	��!w0@ֻ�E��&[j,��L$V��Q�����Į�E�7x�f�J��s��P��<��x�|b�	m��U��7W1O��� 3�f8I�/	�ڸ�1J�xΩ氯o�v��D ����c�_3r��k���7�F�.��)W��D��%�&PY����rb�q�e�#w����b���ƞ��fS�YXp�Qa���q"(���GL~��m -�*�U)Q�� ��9��<	��jBJ_
�����m��DQ�L�E�ʪxv��u�X�cMM�Tl���%j�Q���mJЫ��y�����7o�iT�J�[�$u�ؓs�yQZ��-��Vzx�`��sd�UVj<��%'�2����T�U�Z�ϣM�����,���됅��Gm�������RZ
��eff��`ee��rrjJhh���?�wl++�VN�����F���;�|׵T�,R��{��5`����@�/@EEř֍���i�<|�p����x��V'�ʭʾ{�q��i��K��pާՅ+��fzK�����n�2		��G#�rrr����M7&|.�x��G��\j�^'?�c{{�~R?���i��::�m������l�Kݻ�6��{�+�c�DD�xr�(B��UV��|��j�r6������� چ���߼	s��à$l�g>�]
� ��2�Ɂ��ED��`}�XC�
{��]���]��:� ����S�+�V4����OBBB����z����9�	G�b1P�4r�2V~��A�B�uD�!3|�� |�8y�D	�_>�%�y��+q��y����U���3�a��j����$	w� ��[+Ҋ�If�ɓJ�����g������~h�8=����q�ҋ>&kғ���0$A�璛�4M��]�6&y�q*��;|��&�=�9�����o���fMB���J�*�u�l��fۻ�i�K���Zj˞d�¡5��q������s:��l�cc7��q�� &��������a�^���tO��|��6禵%=��ݭ��ۢ�r���FO4����J�E��U祖'x�TŇ�Cgʯ&[�ĵ:��&
[��1���z�n� �o��!c�,3G2U?���~�h\?�E��aJΓv7*��ƭ����$�Z&���n�?������Q/�P��D�t����y`�����';2�sĢ��<J�<�N����Ӟ{�`�ܦ��B���+����TiC4l<�����D�%x�u����쐛�er�ɑ�ؚ("":i�$~AVgb14{�~ˎ�ʉ��~>��_]�{\g��ZP�Z��M~� a�4��6J�ʨ����_�9�Ӯ~N|��"�l��BUj$�����~d�:���o���N��Hu4�C�{��*f��R�`�g��n�x��cO=�'�|Ԙ:-�~���ؿ����]����C&�Ny�;���+M#��;�aU��,���}��|�2�-V�M>�g��yCc� �W�5|y�LJ�l�,���h��}[����E����C\�J����:���S��o�7�;�P�lܑv�����p�
Ae=]�<+.^l?a:���O���!�����k���r��8�(� ���9�<=���n����zj��G\oڸ
rW{�%*!�}�}2���b�:����,`��A�
}�F��Ud�K����]H�`����	A�]��@M�����}�.!���q5�AH!�p2p�n���5y�,����5gg'*��G��F�FGs����Q����Do=��l65%�'���V��30�0:��� �(9*���j�ᅁе@�!*���Wiiiqǭ���K�����-�	Ց�<�����N�p=�i��>%KK׆�����|mc�--�nݺ@��K?�"-m�@���,�A#��t�����N܊K�����Iq%z����%�ޅ�~��%�p'c�A놳2I��*�uB4��a;Kͧ$CY]��]�51�m�N�V0�ry\�-z�a��@����M3NN�o��Tc�Yv���n�%''8��?7�%��WW،=�S[�G��Pj�"j�-�ʕ3w7���[�w���|��2�Ϧ���)��������ܻ�#"��]_�eu%4���lVݫe�}�V��b����p��)m�[�4�����<�!��jժW�,�^���a��R)AtZQ�<ٓ�^�i5Eu��M?0���DL/���fD����l�����0��~�����a�%��qp�˾S�ul�ŭ�#�� *��QSco"4-.�	���ˢ����P�@�X�6�8��H��:vj������?�D2�z��D�8U� I��u�7R���udx
����R�m�j�w�Q�++� ���,����'5�Iy��el@l�%f{��q�kV���|4�Zκ��n9a�\D��G���Ygb.� �^lL;Q}{}��'���wC��P����,3�)�����Q�N�<$����n]�K5$�j�6�U���硡))�D[��Q5�Rk(U!G0�k 2�R:���v4��D������x#�	-��J#a5�d�Q
��G�������s�@���ÞI**C*��|�Zp�@ܤ�qt/��}s�Bbb

p�f=�mYYb"f�H"������ۆ����`��9L��C�@��y|\+8/C�ẇ��\4�CˊĠJ���pU�����h�V��~�A�-����?��J/����4�v+i'� z��-O����c�"Gٓe_�����+�Zzus�Y��*!�=�9�̹�(o��q�K�:���L�6�Zf�>�h��R���e�~t�Y�ڞ�\4��bD����/(��daaz���2�dUv��k��/���U����v��滤G��H&T����{��L���T�+�a�-57�6�!Q3�v�Ւ|n�=�F��"feVI�Y�����ך���llA��g��\�4噑���\��hu]�7���튉]<�4��ui�^�����2��o�1�p�C��Ob�F6l@?�F*�D6ydd�@�_���5a��)����a�ц�Jp��ɓss��X;���O�>4@����̅�M�ٟ���Ͱ+)**�kF�3l��H���]��p,�k�jČ�O�7�0��%j�T��Ek�-�$����q��̌]aa!��c��)���T�3���}��;\y!o��~�U�LQQ_������a]�g�D��6�������27]��W'C��(4�L��W�J'�>Ӟy��>f�6�����Q�}b��'a���jH�&��c.i!=y�
[f-Z�=d��j�7'��Np�Y�5_���G13�m##jp�x��-��q�B���Y�v))xG�M~g"""XDDD�h�̞+iu�����Р*�d�9Y��{�&=��--m�鶦�X�e0|JhJ۴ixn�'��も-D��<�_��lA�S�e����S����}HK�F*
~"Џ�ěpũ�'��:MNN�P�h�Q�!�������������X�I��x�lZ�ǔ�[���`�Z��,�����m~|��u�?���| @Q��f��n�[�-�鄭X$ů����q����b��B}NrYd�SCoo�`)���8�Æ��B�����Y�f�Zڿt4+�'����s"���6�?�����Y�&
]�-����@��JV��Q������������9�z�ȴ�$j�6<{�U�#��%mpx�>=�5G�ˉ>]�'ff3p�NY(���Ę���`PE�}�l�5+���$4O������л��P'�z�尽V��#~��y>C��ਲ~Xd�s�J�>?0��u�9�9d�k�'�o�	JFC�2#4�`�5W[�h̤F!�u�x`�`�g�0i��5f��`������b�o�����y��T������3"���R5/���bX�
���P�BQ�7��Y�)��$2s
J�H�_�r�n�=Qd���%A]j�<ۮHKC	�����+��>��Fp�Y��+766��]�Е�G�A��.�%8�u�]�?
���[�ƶN���!^��OU�����<=�ҷ;9��FFF�<g�ؘ��=J��WQ���R���-�2ۂ��[k��N�d��X_f>��p?V�1���.b98i�D��W��Wx��x�y=��~+9��4�(m�E��E<�$������z�R8>5
����|�55���'�p\Bb���Jǎ�2�j*F��2�ꚺ���Ǝ��_E��(�anA�x���H�ɶ.��m��`��??|6���Fn���빹k���0��r���82��}H��(���m�=�}㨅�P�o�Ɉgj3y�U�Hǲ���@�y0�0�_���c̥��BF���-���A�u���a��@�<�i6���UVV��)a�+j#��_�)��(%��h=V>.� �"��M����p�e��%�U��&U�یEF��Jc��m�&�,��=��+�(҉������G�ɓ��C*@j�L)R��D��*+���_KO�
>��^5�~Hك��x66!�����Lhy�gd�CAkܖ���b�y�K��qŗx�ƒD�<�""L�����X r��s��_��˟�,����TY��Rr����o�C���S�����A�bZ�X���~�t�BR����ϟ�6��]1���R���S�("2�[����(y8�0�3h�u���R��Ѹ
e���8mE'�ӧO��,A/-��,M�35Z2�^rb���O#��I���M�)� �_�����o�� ��~F�px��a�>��B���?%_���*�o;}�h��X��Rj�~��iTTV�d�I�$Ί�m�
���4�0�n�C�[F��});Sp_�y�f�A,J��.�:}LD�H�N����^/�Ct������+�����`P�䍠�F��@�������ɓ�g�"?x����񇏱1�H�ìYf��H���Xˮ�;Zr��և�Q��nݪ@�ڸ��䡀G�ʅ�6�i��M##��G�Q������9�J�����xɺ���L!#��r�EL��.Ѻsc~�ǆ�vzV~�;�\�W1�G!q>����C-(��i6Gb�`��|�gE���
t��6{{��� VZ]KJNq��a��
�.�E9)� �.�D�����[�W�=��sHi����m�M����j�u㩽v�a�/,ʃ��li�p��g8�=�(�1��ȓq,�#��QeY�=����C�	{�*��cղ٘��Fn�NƒX���P��.''ׄ���8�<d�T�pL)W\��p�"��Y�l����Xt	�0�?s~e�ij!y���ﲳ�U�kqm��#Zr�	(0\^Vѿ�G�v��*�6?�΋-�^Κa���Ō����HK���K**m��
�#dVII�u���4yz�^������KJ��E��ۨ�����%%%%]���J�7���d�l۝��/ww:�N�E���[Z��3����[i)�nh�$��F__�v�tr��(@v����<�/��t_%�C؁P@��&*���˛�6���ΡC�i];r���؟�l���Q���N��21y�u��"�6���X$�p�Ǳc��d�f;V��-�b��oNT���$��R��c9��I�
����=d����uqJ��k�b�<�l��!��1�m�[saUV~�s�r�B�\�,.�K�n���eK��$��1F�zQ6a!�V����Z���g��.]
lF�3̵I�N�\�E��6�->�����hr�}��ﺞ��u0T����$7+"! �H�h;����o����`[9s��x�W�jN�֋Y�f6pY������*�8�*%5ٔ�%����o$%e�Y�OLde���M�g�^TT8��Ж�+��B�W{P�,|/M��/��hR��)�����{b��?����/�� U��2V���hH�8%�̧�]������`o"��fyPb���Y��&DY������� �~�_�.���?����l<�����E��i����XM[L!�|�������}�,��f�yQ]MCԼ�o>�
?AO�-���i"l�����#��%��2���/��5�?.�fQB�.�fسv��1�{(���P�ʛ���RV�`)�94���5�}"�^��6??	>ZR�8��4�)����������F}�M��1{{P�U]��S]�x���1N�*�ak�_Ép�F#;�A��Sb�|���(Z������8;��ջ�v��P�>>@�EO��Ce;��B����\��4���uBB:Wn�P@�PMݧ��VZZ���R3��Y�^���� �c�EA�.)�Gj��P<�
�m0�kyp	�|ba	�
��.[�qq1�k1zՍ��ez�ݱ{��Rc&�)���G�'�_�g]��Wݸϖ��=S����^�<��g��3�pk�1u�Ɇ�-�U]�wPK�G�N�kk3���vx��a��غ����cF�7#���y��s�[JIӆ�Y��zIEo� �b�F�!�+ڿ�I�ŉ�욞iz�}�6~ko���@�v�q�s����L�5�;n�,9�+�K��ܼg�Ř��ϓy��v��; 45�3^�|�%[I�R`�A�ץ�E��&��&҂U��\�ra|{K�7��b����7pw�J�S��eZX�I�V�D�bE��h7  �r��r�Tn.��r{�r��r�r�~�N#G�ⓜ2����>u I��\#� ���Gw�5c��m����C_Ǿ��9��[52B�
���/;��7l��� ����X��\]�GE�Q����K�.�U"��\��U�끵��MG_i��x+l��Zs����*|!kl����b�_�����1-x����¼�Z�A��S���#�?-]jee��d'�`�J4�0�ҿb>a���s�<-,9�g<h��|�j�fFƵ��ˮ��?}Ծ����M���⸦j㤹h���kc�y��Juy�"�#��mܸ�H���
ny�pT�MZ[��n�F��vH���o;L�v(*z�(���g�ܥ����ƶѫ�����C��-D�`SnҼ�Ã�\�U�׭ӿo��J��K���&�z��<�NN�ۤ�l�F�e�G�}��ai}.JJrE���3�T�^]Q��#l?��;�Kiؐ��OL��A(+�<r��Ƞ8X^a~	6_�$==�9��E�UYY��h�k�**>�P��������0��u�yj��0L�{1 b�n�_K���=�L����\����D㼤�ȷ4����2���+�25�ѯ�+��+�����U�{��wm��9ΑkS*��H'--�T%����gS����z����@GD��wJP����'Y��/�[�AV	���%�ሴM
�^�Dp�v9�\�0üK��J��7�k�M����Yb���u�RZք��)!ee]tO���5�ow�M�0�*TVV޻���"E�����t�EK9SP����4���~�uMj��*��zp�#���n�)��=R�_��G�.���OA�94k���a�e�G����T���Á�_����X�!�t߅����.U���~��|r�˞��/gM,�X?ĝ��qm��s�1YٗХ�����y��!H�g4���t׮_ULPr��.\��k�t�+n��6���3Hy��������X��YLW"?��&7�v	W,3ﶼn����R� �{�� �Y��C�Ӟܻ���I�ol����1�����5rd�&.9�;=m�6߁���|z����iC��7���SSSd����az��������ڎ���������9ǧY,_>��e��
v ������t��G]w�Ą�2{4���E���%?�������J�j	���j<Jqj�U���a���z���{k�߮"o/��_�RQQ�1��۰AX�����_�MZ��Jԣ��l<��p����w�nQ?K�U�Z�JTZJZ` ��G���~w��Y0(gPPޤ���N��o�*�v��ϟ�����
�c넛�N�O��`��ޯ���x&��2�˚w{���e-��C��9��n9��c	7K�o�	X?�h�	����+0��G�=\��fD77�E�x􆆞�FmVR�%(u�B�+|�Gd)<W$u�|�oW|=�J�%�	�*� ���~�<d��&	� Ԏ�A�&�>�r��!�u����f�� u�� �����՝�m����HK��H_lE�ɬ;���ӧxi�R4T��5Y��$x/��uT~~�����M�����Ia��^��7�M����=b���0�h	TZx�W�j���y�tϽ��kT�ϳ��q��������Z�e��� Ke�?��oۦ����W�ga.��l���,r'%%G��ߵ���o,����Ƣc"�C�?*�qd�� +�ȵ��h���9�Ȝ|�G��ဗ�R�#nGB�L�l�$�f�ę7*%�-t�|���S�s}2&h��tĿo�]#���=i/Ø$̀�?2�C�����o���sx���g�}�	��̺O�VF����E4n�G?m�F't&����֥��Y���*�";���A,J�jAR��Z��	�e�*�+*0{��K�g�S
)�w�2�A�q�Wo%m����V�o!M�D����R�X� �;ÿ��&Y�6l���5�&`�rU�t��C��S�Ұ�Q���\�*9�o�$�;Cj���	�(&�ۇ��#/�G^�x.}[��b�ҲJ������qB>á��0���г�w�/�\�mt�i�5��>�t��p��
���X��m�	{E��݃�$e�:2=��z������N��Ɔ��6�"qZ�:6��?P�㭀�G+�_i��Jy,��6Ԭ�b0UdXy_w�E�R����U���������'�unz��W:Z��y���iR����P�4�E��Ѿ�L9��Nd�N�a"fZ��y}�i3��*(���Z��>��!cz�:&&�� ��v #Y!���Z�]�>�i�=�V移�]�A��kO�:?����%"J������ht�'Vo��'��ˊWA���b��%�,��g�3�:AJ"w�|��\��g̎�ݛK2�Fr�]%�qq����@,��N�!�711!��n݌�ی�n��L hi� {QU�ܹ|��� ��L |��s&���W~o��"�71�����A���:*�v&+Q�ڳ��b������������@�f��e���H�(�ߝ )_Yd��c������z**?�3��}�vtl���g^F4���GGG�'�7,���m��!�[U%�X䘉��%ڙ p�����0r�:,�魜%V�1bق��
�u$~��y����Z�⣭/<&h��3�{��f����9�Ms�b����@�L4��u�>���F��z��]pT���(����ٓ9�̫8�+3�Pv	�'&6mjI��TQy����!���3uc�,�7=�έu�˦�?�7�,��	�Lc����i]?�<��5����w�\��0����qp[[O��,m��?�*��?���]Th/^d�4�����-~!#���zy����]�}�)���[?J�M��?|8���;���7��ȷ����h≵|���5�����4�U������k��������6�.��n|2����5�a?C�r#�&K *c�W�H^A��4m-�*..�/(X����C&Ѩ8ִ�K��C�L��A�Ă����3f�Ie �"�AuL��\>|(x�ӄ��_բr,�
(hn�(�Rxk������İx�w�2�3�� �"ځ��f�M�̊.�U39!N�t�dRg�_�2L���zWr���P���������H�
'¸W7��NZ1��1�ӬY5�k���Р�؅a�'O�V
�$m��fg��!�9�3�u[����/ϟGD���Z�4�9�#���C�mY���D�_�����%��^}���sǶ�ѯY������ͅ�,Ϝ��N(���b*8�@����;{�������
=xN��S�mF��y-���|�t� ����w�����3m'��9,�"�U�7ѧ��B��j�����S/�M���.�:�>>OY|v���!��%t'm��jL]�����Gx�saWV�f??S�Vӟ~�Z�M�`�C�z	�?���&7�&�������4pJv��ٷ/�+��{fs��C��7?��=i�ҋu�;�:ʒ�,5v���J�������;��w��4�T/M��7c����²��g��G�ak\>I��ϛ���o33���~��>�>J�P�Fj��V^��&��e��
��-G��Ď��]p��E��~iG���^��ɑ�`A���+V)-e~�{��[Ҁ��	�Ø��n+�����ճut|���f@��"J�^L�G�fg_����&���;:�T��Cy_�o󸊎������0|}II4��Qw�Ñ��t�_�v�_�ھ��=|&�����X�&ĕ������IaL�";7��ݫojJrR5j�I��L�7�foo����J7��U٨['+)2лX�нm1,�h|��3k��;�.��e�oH8������)q{B3$䬔�f������Z�m~w��?��l�R8��o�dm=����po_��c�5E#G0)9Eܕ�]����f�(����u�Eul��FcbK��3!�OLLL�$1�%J�Ũ�F4�����I�^X��Х�HU�X,K$vAMb��h�����ݽ��������|v��9g�޹�ܙ3g������ۯ6p�!�ajn>*�]���R���&5��dQ�����b��ו@"J�o:R[�����9��|g9}�a�T1�[͇��[/l����|v�ߜZu|�j�Q}}!���@��/�7w��q���?���k�������}��M�w�,/��>00pi2䷽�vt�9��GϽ�7��*G[[[KCU�:3؁F���y����[ZQc�ʅ�9�1+E�z�R����`���Y
:;�Q= e;�Y��~p���fK������t�R-e{|�Y�r�u�����R������x��(L�;3[�oo�^c:�+YBG7~���b�Q	�Ӎ��&�˾���x������
�a����̮�����D��^�?��|&\B�_��Piyr��4n��Ğ<�p�4��G穰����{�/~�X��2v�q��=u�k�~��W2K~���˰�$b����֥Y]z�쒵)�(�6�)�Dfl�t�Y˶��?�o!��:�p��76~������c��53{o�r�G�կ�{��X��<��]]{�۲e���RHWˍ��<��ai���g��-ם���}���J���c�^m�3?�'wE���|}��}J��^Qy�Ow�l���NД�6d,R#���u��/r�<x)Aa������f�^�b�9��ҭ?aT��p3�L3���� �]�`��O�M�14������a߯�!S��n��2�����<殄�AOLL���[6OO��\i� ?)�\p��N��N�2I�b.:��R���%��1�܋�L�D��0@���%z�Et�v}��������~���t�������tw��4b���)2����2�/얅�Q�ƶz���1���G�P�{���4����;dm���+��ss��{�x��^������z�s-A̗t��]۱�ԣ��U~ܥ����(�nf�#k�s�������������?̈�:o^y��1�!g״����?J\n3�=t�Gۿ5@��sP��TN��7���i����M�9s�O�F��,\|7k�#�)�%�|y�b�ls�h����o��zAa=>��tk2IF��sg\K]1D1+&f�u���Ir6v�ι�������"TW/�����l���q��k��ާ̟2��z+º�~J�)��.��b��x�f���x̨f$��'�
���T����c@�{�8ي��K����57��'�h�ڿ��$5���"ϻA���B�)t�*���%s++����,�~�.����mA(BUc�y�Fq;<{9[q#|Q��ر�Ϯ^Չ�7F��McLA**>�������3�(�!��<Ȥ�A�Þ�	a_Ū����=m��9��گ��0�c��Y�k�ʸ]F־���z���W��ݱ��A��O��훺'�?gZ<!�sz����G��n�.��ϙ�L�������5.�w%.��!�ɝQ��w�?b��tZ���B�IKsu�����묻���?�֪��W�s�:�}5���}S�԰}S֓rO������		/��޲Mo��>,6�*O��N��LM�.��n�|YfP��t�8������U_�t��N��kmf<6��3�r��#¸p��yS�>�Ѷ����b�����3�_�u���}I�G��8���>����X��F���յ�y^)7�ΕOzgxO���y	������+���R�����Uk���h��_SL���.F�뺧i%������n�p�f��|�9��_�e����TR^�RZSx*�;F��X��/OV����n	��_��1�Z&"¦+V|���~�l[�����cc�o|~����<p�_�nx��a��E'��K|�;�/�п�8w�4n���/���RSi��؞#)'ee!#n��saU.'��~nLs^f���p����6ܹkY�A������{֮�z�����İA�K��ވ}��(o��!��ӿƂ�g��H�Q��+��l��v|3o��#)��b���
��{g��LL��㸌#�%� #�b��B��[m�ȸp
��Z�����3_7�0�=��˫����&����F���H&&���(��1Au��rf�z�Ǎ&ƌ�?Lx��G�^��(�|�(�X��M���e�n�?�71���Q+�7�k�z��S0�rI���w�<�̨���ۃ�l��2��2�y��u�)�,��ݼqcfN�\ce ��\��ݣ;�1��}o��?��<����=g�-%3_����FUfw���nŴiM�/���4��{�l=�?�N:�G�P:N�T���"���I/�A?��P(�����8��Ng�4��i<�HM�Qo0}�듹,̼�'�*hj:j`ph��	ם������ɣ����~g��������1��9u�Mv�X���vh(H�P��^�u*g�`����q�f)&���hb��5�E�՜��e��m�+�@�]��խ��2�C塀�k)Fm
]`"U�O��zE����w�yg-Xu�����9�Ț���S>��X����DU�����G˺u7"m�|ֲ����Җ�#_}s�����5V���Xsk�F��Zڻ��W�ڗ>���}\Y�C}v0�@^�p.*"�g��̙���s����/�g�#(��Ef̘�_��o:��>x��Çi���1n�gN�hgg�p�����K6�����J�G�!�6j�?&�)�*�]gz|݅?���`��mmm/ż���O?]u�����{Ư���k�xD����7�/�\9�۴/���/O]�38g�&��_���x�T��2Dw�<-p�J�Md!e�* d]�[r��'#M`�̡s��UKUw[g�`�cثpZ�j�Y�㌌�TG^�̌��:�+*�F�+W�<�jK�jf��|g�(p��z��c�J��*�Ç��͚�.2��&�(�?�8����D����޹���MQ&���	�"C}y.��^�0��;�́�L6�QÇ<������t��|w�B"���w������o��w�0F��l�X�"�Z���ΝUU/^��~A���:D�Z�سb�����A��{��׌��!]3�^0�+'.�7ו�_.��ә����eNB��d�������$�l߅�C�oJ.�$y��`�~�)�L(�K�2�[�N3qf�j�p�(�	ط�4$n�Q2I
$%���@��	1Y�A�=-88��%��yIF�$%<lo8M���XIV(�r���XLϐ�����o�������i����<2�_�S���DV��@�$+#+'W,�/�//*.))-+W�8tHQZT(%�R�Q�(VTVU���աJ�ȑ����9ijR*�Eo<g�4�SS��$Ħ���u�pƂ܌xt/-��QqIpf�X�T�/�Je���K���G��YQ>jHH�0d�2 �^tCuS2q$QAN��sI"u�)+�JHL�sp �"���EE9"Qf6N.e"�ux� �7_�dJ �J$B���EYrD�ɋr�EB�k+?��=.��c�1��"1�L����R�Vq79��OJI����H8�h�Jc�s@ЃǷ��)Qx甘��ظ�da:��<	<�����_Z
�4�4��� (T���kkk*��ƺ�� W}�`Y��@����v[���h�e��3��/��L���	����C��������� ���X��j����?\[���õr���XW	�ƶ��T������@hU5Ӧ�#�h�Ɔ#u5H�岜T�sʁ1���q!�f%����xZj����c����䆗'��,6����Wih4�%��.c6y�7�;_p������;w��'�S�l��]=\]�����N�<���������c��n6/�]p�V���@�KWjfk��l��ykWX���>xߖ-�K�r�|���;��сbz�l���OĔM��|h��yX|�^n���nid��Л�����da���w��"n؎y��Pt�[���/��.����=�<��}�r��	r
���G��b}:4���8�v����3gϝ'�BΜnke5�Y-t��N,t������ŋ�.]�|�
�^��F	�#�ׯ^�r䮰B7`�D�|��|g����jq��)�_R^Y{Cs���Pmǔ�rT1���	Kɖ()���=�ݤL����9(T�q�9��4�׵��l�A- �O?s��ٳ����[zR��p	�齺H��eD���s�m'��������[��&��-�c��������
pϲ�-��rE���Ѓ��
�\��q�\{�ExR���(���%��ۖ���L���`,,�LLʑ�+���BXJQ�,� �����G2)���Ĕ)��J�`��hQ��@�8�A�9�Gvvt��
ڎ�
˴�<���Qj�:�;w��`���'#26�d�j�ړ�����p��͛�â����esF. �T*E�� ?rhp�šMi�I��� hk��ԐUWV(J����,a�6���	���l_�Hׅ�X�I��)�K.8;s��==��I��.��g �8�^�>��幛/F`IX~�a�����dȀ���8LR��"�X�	�w�u�&z�|�겉1�Jc����� ������9�Q���
��9xy�md��K`������I�9� �u�n��ƀ�W ��$�:z�o�^=@|A��x#y����"#u�'#�L�Ʉ�P?d�efJd�TbiA�HV�b�xS�3*A�Ϙ��ED`��f�N�Edط��[�����?08lotlD���y����_@@ u���#ḄDAr�0-=� =%!�.c[|B� 9U�/}�9X��;!A�_�4�y`Aj��x`a!2	`YLܟ/��e�&Ƅ�zQ�]������A����$&�Ǆx�� �-V�N�<�� ���	���N�L��Щ�4�c)xFZ��i"01W��FZ�}���^pԮ�<-�$���,���9Y�tɉqѡԠ��r ��\�Ė5����ӰN]h�sM���GB�s�F`OXh�-[�y��h�B��N�e�ǩoUb`H���Y�@���G[���7�ٸ��DAD� ��k�$����{��	�p.aI�1⃜7m��K�n܅9($<*)nt�#���у]��)�XÑ�����nnVF
�� ��-��a~ĦS�F�
x꣔C+H���Eyy)�ˀ(,/+E��P����`w�����ݮ^�A!Q��K�'�BfjBT Yԝ����d�ЇV�2R��=����Ut�y㜌����-�;�N?�V&
����/w���P�oOB��C(L@:�YU��'T�nr ��Y
�]O��������܈�?V�h��xqB!c������j����4�L:��L�f�Q3(JI�кS~{I;����ΐ��u�L��.�B}'�:�X&�e� �E&ˊ���:.C$��rs�	;wu�&�A�2�.� +3p�{�ڕ�;����}G�+*�B��v�s���ˋ��d�Mw!�ݻ������WL��5pp�``�m4� �z�5،���^n�$�����^�֮�<//�>���B<D�e����Iƒ΋p&qώ(���k���V�%x�����v���,,@���[���Ӆ��0�"����E%ea"f���;|��.�HѨ<�}s����	0�����*uǥ����v
��`�ck��X�C~�]���p�tz@�%^8��tǩ��Z��5�����N��P1����W�.R���\kq4�2��=$$!MĶ2Oi!�Q����Zd8b�C}}]Eq�z!2�xaV�1�ť� H�(ԑ��$uF��E�R��8���Vv@
&�����A��3�J}B�}㑪��fo->%��s��cW�^?V��1���$����7m������''*��1�]qS�n���u0$��k��T��=]7��t1�d�V�Sbxp�]�,]�����J�׺z ������������j��o�Ѭ�[���TsOOt�st\o�ɜ ټ�5P�zG&�g�󑆔�^�.³�ѣ&Vۏ�~^��M�W�j6<Cx���a���G�)�@8w�����ʞ=�y������kE���Na	iYHA��A��^P������R�<h�Q$z��9��	��&W�#��:t��mh���CS�S�%����_ĳ\+�G�
ݔ��<�Ç#����K�����:S�q�O�>'x�2��]y^��� E�����Ҵ�_2�SC��+K�ħÀ�@���a ���gM%�?�Z��ա�Ф���) �h��
�I<�G������{n�Uا���yn�_��o�\�'��$� �;���A|4' GA�C��|��_�I�Ǵ�8�D�<�h8�#�%�0ٿ,��� �I8��te)����{ԉ�&�Xn_�'����r���e�/�c ir[��y���O"�����x�r���BW�	xR:L�f�o5��t\!p��X�ZJa���Կ��߰�<{)ȭ"x�L���0�yT��<��f��i�Eh��$M��0������P���Ԭ����ᔧ?h�c�Ij	5XQ.��A_��K�W��R'���@0�р�$Α��<�x�b��(���<o5���Ȃ诮�����&��<��d�9r�\PɁ3{���@)�,rC�TREE  ����������������zM                                      ��                    f       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^�\����/+�>vdgefD��ICTJC*!�h��&�e��^�Bf*��QY��d��s�}?O>>����}>uޯW�9纮�<�<=�׹. B���cxퟷי���k���oJ�i
�U'��$����*] dhX{n)B�V���[���1#����WZ��BiiV��K�Ei���\�wIs��|AS��V9�Ni��(]Y����P{�v���P��$5����c*n"1��~^��d��6ÏoE3�lNr`�=U~�$� Zؖ�Z7�S�A��k2��,���J~�iSe�ʛT��U؜0C��-V ��fw���3	�����te�X��W�8C��J�kū�e�#�q?��$ާJG�G��ѵP�ڐ3�������☶�r�%����T��9�z�?47�,PQ�1+����TF�����9z��D�SR�s���RC�;��W)���x'���D��6�b���LU냗HE�i�m��y�����
��޻���ac�S�o�}3�%����0�E�l����s}��wjj~��R�=z��Ѣ����w�e���Ejjj&�j�\�I:�n����͜�jKWN)���"�-��qg�(����h�]Ksn_Z��y��\m��ɢ]=L�\��}�E�&�F��������[W��.���+L��Y�Z�FM�Ԕ�IM	B�
X\t�*-���g-�&
p�ԩ7��#PЬh5���y�i�+�6�G��bjxY.��B�!�i����� ;��5�ɵQ�1lU�n�p��Py�37e��NV����m}���"k�rg���R<����`2�F�dpn��T�xEE���2K�զ߫衽;�GE��i���e&h���]�G~v�J0U����lZo���%��&6��^��p���hp�����((f0����R��YR�2��h��ZX�X��9���+�:�� �H(1 0	�J��l^OP{�*�nХ�m�TVF�<܉���v�Z�D�u�*�]L��K�Hs$Zec��W��4�^�*��!;5K�u�R���c�Ls ;K�SQ9rw�駰���=�1Z��+-k�ƿ��J�j):�tQ)�NM��[ �8pw������}ORXN|^ظSӁ�}n�]�x��!00w���|�ѿF����c7��ݶ�耓Ј���f)I�G�,P�ٮT��t�[��U!8?_���^��NB��U9���hbH��'[�m���XP��H1�f��̹�Z\�@���ח�ɵDv��v�U��~CN��S/��,S#J�3�V�j�gP[���!˪3�Ώs�R%Ϝ{R	K�Ν����+�Õ,תӻ��}w�I��e:e#/�闹��AW�j`�6T[�$����~՗}�[������\����JKN�l��5�bkw�k�]�PJ�~�{�x��ڵ[���U\°q�ڈoּx��Z�Vj��]�|Ft��֧k��\�hp���i����]���L������Q���;��9�[|���~��Il���]m���K�KM	�"�n�j���	w������q�w�6/���W���)�1���S�o[���:1)���=�4%~����a��>6J�x�8_��E��.��aF����2D욋j|Ḵ.�.yc�4�2]���@]]'F���ۃN�R	�i�$�`�P]�.O�PKk 檞���r9�S��aR!,�{� �������z*��R��'��sq k�ƌ��=�;n-��.��e�*5�ڛ���I�%mv_�Un��=V0��1'&��Sf{��
 ��`��0��R��m�	^�;GR�o���
}Ī ��z,�$�*]nJ_�쩋H������N��@ ����_=<��I7{0]*e�i��=^*���R���t5�Wb�d��B��S��0�)��R}��̒<65��ƭ/��S�!�� ]D^a �=�z���Q����R�@ T6殻6u�̤��š-��h�	Wz�jo�z�w/d1�nT�PN}�i"�����]��<�/.�խ��1�q`|��ϒ��x��ĸ-����7%8�H�z�u����1��}��M�M�U�;�-�AM T`jLb*�1+��P��x��Y�����+n6If�	�O0�u T~�@�ch۝=�cNam�J\D��xw �_����)T����iq�s�r����S�j�^ߋTU�g\�g�|��QB��n��n�×�U{��=���vdS��绪�ꨰ�������S�">븩��"U3�A	�q���ꌑ;:�D|�ꊤ�v]����U7He��U�}����ʎ��ٲ��6=pQ��&�?�j<�W�r����wEZ�,3o���$R�V��p!��=�����@@�V60�����׃�xR6�����߽�z��Z�q}�Z�I�@ �_L���O����511q������?�R��Aꩄۭ�O���81��N"G��c�֎�E<2Q����g�Q����-Z�xN��B�t��'gi�^���z�����-�ن��6��M�� ����EGӋ�d¤����[Լ�iVG/_ ~���pH/k�sB�	�D�Dp��N�Ʒ|�P�y_���C�_ ����*xD���Th�/��ܣ��d��_m6y�1@�*+���F�)�"q���av5l���r!�B��~��~�簀�H!QoV�!?8�b�$��P��_.�!\XJ�J����Ə� r�[��o�X���RY�=v��F2z/��Nd|�����@�(�FMp�к�֐j'X�N�5Y�4T�/��q�� ��!�D!1o���:�m������{�Ix�����6o�o���h�*�fH߲ウC�yy���#f ϲ��@۪�ܡ
@���%J ����P\��ѩY��Վ)���V ?6Q�(�3�F���
��px;*�� ��k�pP��<���3
>ͅ��(��*F�o-���GtE6�+��f�z����
��n��J�t+'@>���K��[8'& E^EKo\��� m�څ���s���=�k�M���b�?���J�哝PN�5�u�m��D6�����/�P��?��)�[Q�]��h��O&�?��K3� �ȏgRa��]i�K��؎�L�e��/���n8d(�G�d�e��G���$A-k�&�I}C/S��dB���(���k�pÆ�O��>���z�/��L���~�'���R��^�ဍ����Շ�/�'�� ��>'X#�2�YG��O������yq�F�ܽ`�վ���K,��z]]_nm3����+�ܷ�e�s9m�q�V���ŷ}��؂���\9�,Ǵdm��u��ND��Ʈ:���L`�I��beu�~����Q���h�}<���
�p%�zN�8YP泲�(�[6�ǟt���*�s������x#�P�i��pꬱ'��՟�������]�����!�vo=����ʵl��R�`y�w��z��p��'n8߽{�H�C��^ASL�0\����a���W�;`�>Un�5μ�N�/�@��lf�P�ǀ���W�_�$�ylP�;:f�Py=d8��1t;������#}^I�sj�|�>$�ӛq B	57��������:�La:8�@���X���t�ˊ���Q1�^,n����oPJ_Cq���V����4�}&%!"v�S��o�OJ�|<)igJ�!@���?&%��]b�G��p��"�I�A)���5��Ӷ�jm|���� �������-6+㦾��ˍ�P��z�a͚�u//�l�O�8..\�V��)�E芸�����uj�nw�7�s��+�a`�@�}�b��#q�Q�.Fn��������&�w��k[:��e����߉o�P>�V�V,��ym��0D�<���e��=ɏ��2������5��6u���|��zG�t�U���3��;u�z%K�<	m�����{+b}�(9'��m��������.1���b��,y	O�k�U؀��-B�>�i��<R]�?�i�#����w�*��m���5��G�g���X��Z��t>xw�%����ߧ����b��0��m�֓~K[����I�5K*aX�++����:�z������Ak���Tq��W��M:;�/4���5�_WZw�v��I�v{w�����@�p,�����a�jj_ܷ��߹;��~8�qء�(��9�|�c/w�顅S����f�i�K���\��ۍ^6�)��=I�����vob��{L���2���o��K=��o v4�nI۞��撦��K!�(N� ��~Wy�D;f�1蓊L�z��:Z�����u<
�s㩙?��^��b�[�I=�v� &E&��4Rn:����)�3�V�*1��������z*i�N�ǮpE��"^7�u�����1Um��g�)KU'�xI�ײ���T����z(p �qF ��QGf�0��I�va�+���}hP�s�@ ����U9�[+
��2��{�	RI�(<���0~�#Ѹ��Z�f@]Ɣ�
�&(� x<`V3�t���c��0�0g�
�����MZ=@n�茘tjE �
�O��׃'��/��}wX���I�(��4]�}�~�7uЩ	�r<�������jKw�̺��υ�;>�e������z��w�Y����7%�I�{_C=�t�9���9<kXi��G����k���̥B���� ${R������y�aK���
�*0�h>��8j|�G/��11{ލ1�8"I��=�v�,<:A�u���Ƶ��������B�R4�h���2�*N����#]�,����Y�|��XY�
���m@�E.����~}���Ϻr؟�k��kkaqB�AiL�ǜ'Θ:}�Q�Ǘ[��6�QYO7l^n�.�q�<����?��6���oT�M��pq��������%T�Kg���8�p4R�Ѝ[��!c.ϋ� ��A11����yh��!.�cx��(0E�&/�x,�*��woǄ��B�	� ޒY��H �@ �S��?	*.b��������5����gT��RT<�1�v����=���V5 2W��hK;C��i.��C�V{��Z�����m������R�8������NR��fgZr~�س_DK �� �D�PP���B�_"�:��- `ϋ��� w�z�0Ǉw�XcƋ,�x1�r5�q�	E��hL���j
�wxڟ��\�*��z�V��r�� �������-w�N X8��N��h6x)q�ɣ'R	��D/9(��@��a���R6�@w��.62�9�ӅtH1�T����4�@�pV����J��5�z4�z>�FT?�(VH @���c�ҷ�
c�Ќ�D1� �PquHW�O��)�ȭ3�"e, ��)hx؝/s5B�pͺA��P��"P��?&<��l�
:q>(���c�N�=T������p�4S�n"h�h�(��q�愶'�G�>�,#����`��-5��lbӨ��@��1ƃ�߰��J0Sc�n�qU�ȑ��`�����Y4�JMׄ*�������h����ߨ�=@tckY��-v'v��
�$�d2˒��J�7&�/��9_4Rei9�Qq��9Rl�W.�<G���y�"+�Ⓙ��K�� �G��Sq�&L�l`���Tٺϋ�Pt9.��(_<���
�t]�}�TF��|�-K���?ҫ���V��Je�[T��Ћ8���uLI���T��g�]W����Q�`��p��E�u��#=�"*@�;��ki�"=����;"T&|б>Nλ^���ܷq������[�}dx;s�̃6f�Qp���׷fW_(�B�3�4���0�����~�Ӥ����N���+=o��/=x8��R���1��r��E�ϽU}F��~�E�0M��ձn+	��H�V,�v��7�/j<z��W��l����ws_��RS�PA��:6h3J�X��^K*`�ߖ�~�^�(��x��(L8����7��>	`�ux-�- h���Z�ua���I8�4C�>l5e�iB9B�dȱ���j�A��tK>7��@y��׾��D/Wꎴ��m��B�!ٵj��U�hi����f�cWj�*�=����!@��+V���/_���Ҽ:�a�kg��zu���qRC��3�Mh,�n����ܕ���Y�����cR�Y�'l>�dK |3��-�ļ�Mg>bmVS��1�9JOP
`cjq\�wv��j��n'�q[TP��8���'W�4��5z��c��Q驓��(5�s�E}8�`ci"�)�'(�<��������vO���N�,}�����s��=�o�h-Z�����WԪn�mc6F��~�Rbx�K�2���j��W���5�B�YX���}�������KL̄�116��q��g�5��+bl��bb�Ule�w	8xc�Q}���`��&�������?�it�=�<�����fO�?��i�J���ηE�� y/��SU�#b�O^h���Y���׹1����kc��@�Ѽ���Wy9�����3Nl�R���Qt���f��8,L�\��r?E4�nH%,���b���7:ld�㨘Kg��4y���:��KxZ5�1B�Y"2m ���ֻV�64�uJ����hW�;�������s����w�mo:�Ւ����D�4un)Tپ�1��A���Kb��mG�Ԙ�c�Sa�ha����e�Y�b��^a۽�|���կF�3ǿ۾w�q��qUk�)"#���9,{�Z�k��}����8�z�jm_s�p.�4�@ TD�R��{����{\ܳ�W&j�/�pB��m�B��x(�$V����.6������,�m䧞��ALyLw�l(�,�Z��xRc�c7��5�l�A���SJ����]��	��<"~$5�Ȩ2���4(.��޸�=��u&��k��x�z0�Ck�?�RK��xˏ��f�Z���T���XǴ��`<
�F�A5��6wD����~�ʊ���D{h��oRkT.�$�jN���0>��S(S��9d�v�u냍��|��P���Nͩ�3�7�<˾�@@��0ɩo��ē�����ڂ���a�o�����q@`	�~-��?J�.	�@ �n��>�u!��~��7�x�R�Zfbo�U�yI}f�3��Z�AhB�NP﹌1�����1����� �F��R4F�z�1��$0|+��Tm|�**�;Z�K	�Pqx�ut�=	nG[h*��C�O��t�JL�m��]��-��Y����D\>���x�T�[����8��gMͯW55
�u˾�wQS3�%��vH�7]�)!T7B�e�2�ե ����6�h~�N��m�{\s���	~�������saX��)-
�{ތ'>u<��;��ѧS9��>�}��k�(G��q ���g= Ê�V�q���[�b"ba8�Zm)s~"�y�����-`u�G�0~�B1�PY0��5Ps��Zqf�V�M���?&P���M���93$�<��n������@ ���׉7���`�o�}̊��	jtC�ӗ�QR�I���>Yz��9�s�xX��h�6}��JM	�@��rcb�P7�Kx�2 �K�"�\�v��-���f�+Y(�E��S54���� p@�\�җTI��f��w_�-�h�Z��|$�)B�@ �!�t������yX���I��TW�龺��{㵖[?:����dVu����y)XH�T��e{}Q-����������P�z�=�.�N�r�as��칕HW��##�8b4a%�Ǿz=�2	)x����v s����U V��O��7�����(�	CW�����-B%&g���l���� Y�M�j�8�y�t��7�I]GL�wR	�e6��^����dL���}7�KK B�"_igr8�M#��O���	BE���F��L[�:��R'��ίs��)��0�E?]�
 ~HsWܖ��(�I�~IdP����FQ』h'�{~�]��W�p"�?&d4�'M �����N����-
� N�j��\�$�';&4�W��d�@�`�� �Ǽ�y��	i &��v��ж��SdA����F���0�����K��}=�M�������M2��?p� ~{ڶda�_T1GjZ)Y~��HV�й-Wyaׄ\.�o��q(��Q�A(\y,'��׶�G5uj�#;w��w��3K]e��Zwp�.EW2c��'KQѼܨ���ҕ�y$T44�1��r�^Z����~�(�iD�~���
�y��n8�|_P��lc> ���u�ᬘ�G�z����/�b�h�(wT�;�
%���߈z��$]V &|�i�m�@�0�9�Vك�Ϡϓ�Vk���R�S�I�} H�(���k(���.?�.S��'O�a辅?���l�_���p��/�g�,ڟY��7pI�k�H���C���_���F��"�G[���6��+�ֳ{6�4�)=����]�{�EGGm}~�4���N�D#�'V�n��x��o�O8fA�̅?ތ<v�Aҽv'��l��/�n����R�~E��A�0����]'����ʡ��wiE���ZCR��D������n�/�Q�Ԕ@ TP�{��d4�70�XNY�uRvא���V ؒ$�@-:��� �D*�� s2��(X#�(@�M�/di� ��gg�kڐ��I��!�*/cM���}T�{G���ʃ�������(�d�9�?���u�z{����������v[aRc͸՞���V�{��Vq��)fE9� )��~��6*YTi�M��0t�>ͭ	JNC T.���j���ԟ~�}p�����	L9��RSTv�"�"�Q�,�S/�|ͽ9=G�U��Lh)u�� D
%� ���O� >-
��5�bg�Ƭ�Ŭ����{4���@K�rx73ݳ0&�Ɠ�5�Ď 	�C�~��o��wWb�g��?�fa�����-�\:�oaa�\�%� 3?9w����p�e|e���I��`˝qo
=��S�;�~��F��飡�QC��E܁m����2�a��Ihhm`���K�z@M��Nm;{]��f��F���nM(4�p���Ŗ�0�W��Ab��X-��'��5�'wp�1\�c��>�q>Pݠ�9��/'c�˳N�}���}��%��m�^Wix��:��~��/w�.�����a8��w�'/�%�C
���B̿��r�Tɓ�V*a�~�Pۯ��%���K���]o��%�.1�j�����N�x	�b��}y����\��n�1�v�]=���e��hc��*��mY/���\��X���m�J����GO��l�ԉ���}��U�F��%웱����)�ŭZG��.)#斳@ ��k�=�����Q�/��t�8޳�3c���k|S���Biw{-��L�Ux��j��ll;�Uϒ��jKM	�"�5�0���7�;���V,k��p����d۝�:�f�W�ؤ��yʱ���� )�KfN>�u�_]�[�<E�?�ك��,�1�r�I&�L ���1Z��-C4�X��o�Q�!���T��J�scϔж?�)+�e([��a�&���ri@s����]֪�$UA�I�$�wB>)������y��;7� �G�C�~�z%��2��
kJ����:�x�8�wy�*�F�]��)�SK�G���F�l%�YH��sI��14J1�`�{q��b�5 �}�����ۊ�;�0�=5��� ��`�̧�l��(���*=���(h��TG �@��"HN�䘐2e5	*�2���l��|FS���X�,�kK��B�����H�@f���hL�ƫ��'J������gC���T��5f�V�{�4�4V���Y�`����@ ��!e�L
M4�T��t���3*�u��4��!g���2B�_߯���<r����tG#�ŚFFcF�b�4��[�K������;�#:��������p�ĸu��5:��3:{m��Q���s�+pB�fHS$�sO~X�*�_ *wU�;�/B�6}�Jj�4�1��L5��M�L�sj�؆NH��)��6j�*\��h�Pg+���O���::�rܬ�C�Bd5���;�O�c�ʤ�&*ϣ��w���4{����>N�(P��������y�ol��M�m�Av��R5�~0���9��fu�˳7ĞM�_F7h��V��8�w�s�rhY4�Tw�1�M-����^gq�&7��@ �I�*ᓢ:�qZh,�+�xx_�c�^(�;�;�?�}#�hЙS�S��8���3�v��s5 l/@���[¨���1c�`Ժ�1 JZѲ~��@ ��7'5����Efff��+�R��o��AB��6U_�X�v:33�-�c�ށW��;�"�'����^Hë���i=�f��N�*1�GKE4� �Z�t[/����n.d6�%�4�Nv�%�;��7�Z�@6$�!�sX�n��-5j^����Қ��ư�1�KX���bc9�&�"Tb���#��/>]a�`�����G���XF+Ӡ��#w��:w���k&����¾�3c��b���|�^Ǧ��%�	pO *]�+��"o�)����\T0
p���:�|e*�:ԣ�c�8X?���eų}���ϋ ��B/ �S���]I��J!0Q_�P��iQ}����W�@��<�ۧ�!���������9ۧ_����y����^������-5�$U��>�G�р���?���zШ����کQUn�ߋ���뛞*h�ȴ�vJ����ج� �;�xV�mV��
ټ��67D�H
'��^��Ϩ�j3%?D�>%���(d�B�C�����6��f.	;��m�rv5pB�RVᮅ��c�];��'#�+5�1�*�Ce󬜝���r�@�����}�@��4�h������@,�p &4Y�n8d���O��A�u�O,������CF����q�u���LT|��
Ĭ<Z7]*#T^J��
�V�z�T%:� �ܷ�Й�X�Pv���g
ڈkM��e��j����|1�w���g.b.?X�4��	�k��P:�� �ǈP�0B�{�>���g� ����T�����I=�i��|gZV�aSSS�I}hBGj����<Q{Ŋ��w�M���0����gZ��*��8L���ͽ�ۼ
��>1���Q�[��J��K�0�^���.i�Þ����Po瞯~/�<�m�,Z��:�8��L��PA&��UnuW�G�r��1��5�#e4�ls�I=�E�z�Ie)��aJ ����\��h�EG�î"l~ӗ��VA#��0����UnϘP�y�]�f����*-�l���%P�R�0��-<D��^艴�5��)�+�@�o���j��ڿh�ࡦa��ڪ��i^ol#h�	\-�#�isghf�|yJ6�n�P��1�:x��@�T<b���}Mj�m�l'i*'>>�2T�J>(�F�K�����(g@b]\ߊ�H� `�l�#UGI� R���2�G��a0�����p�r�%��MED��+1J�Ư��I�2ZG=ڻ��q>�:�z�o�[��!q���n-��?�DFƺ�{�zR��<�1}22�<�]bZ<갬�c�-��`W���ϩ��n�m�bϹg�z�(����;�	0������G�{u3�By�v�E�֫��\����aT�ng��(|���7�9[h�]խWnTT��W�K�F�_<1��]��H�@a�� b�՛ު7�:�<ʶtv��u��Ytg?-CD�7���ŷE����@1���a��H��KFND�m��W���]Ev�S���}<{��|��	�}rz���9���n�[~T��`�|�iBu��R%��r}����띿>��ʦf��
��;�ڏ���ߩ?�G��x	ϵ!�N
��M����9���U�)�w>���6N^]�� ߞ<��a�5f���T���˃������ƥ�=�ĺFA��%�Eu�8L��$k��\\K�[��+;{v�ƳJ>��hz�O~��Yճ�q��Cs]\���&�q�9t���/3�0�p���Pl�o���%g�m��BEB/a~���[l03���`^��w�pZ�����e'��:�Q5��[�k��6Hi6+�����[;�
S���鸩p~����r��*?�YlJ�#	e��ib}��M�%F�������VG^��X@�Z\53[bif,�	�d�U�4#�091Z�ңG��7��y�q�|r^2�E4�w.�z�r��q�=
�p+a��ߑL����֬HȒ{>X� �起_/JW0H���w�ɻ[hn6d��e���YAo��K��?7��`��Xʀ���]}��u
8�\� Ό���+����{���r<��e@>[5W��I �@��pV^�2�V+O(H���<csbs��S	��B��Sc����&�в�� S�G`��u?�>�7�2�D�%zjn����ƫ�bJr�` k��Zw�)�(�@ *����}�N�%�<s�ު���)˶�};p�g�Cwn	�g���D�ڎ�k�n/�K��)>�&$��������^��v(���.nlu�)!��x�i��P걶��YG��['4~��I��IP��>+�'*W�juDb�cC���[;X,S}A�*f�( ��U�	��9�>OԎ�a}
�E��q 1��ԭ |���Hh��Xt�4c�(U4��{��п3�D���!j�MdN�	�����?�*�5S3�$�����UM��'�����䄌+bS����y�̼QH�@9� ��@r�Z�{S�=�b|yHz��V�u�醀�&!RG�2�4��9�-�wE�M��i�\l��|��@�W2l�sT����^�H#��y&�P���!��=�d�p�s%G,Xkli�!�2���8φ*@	��`�Q���[�)����M ^0��H�9�@ ��O��.��?`rrrr�%�!m��vJ[�N-���難��ˣ��u's�9�p���E^�#7��/�`d_��6�j����S�ZB%B3/K*�pz�N��v�Tq9���7��Ɣ�P�e_��^�����'2�V̡��Sl�(�e+�

�VG��4w�A�����s6y\j\��A�t\?��������3�Z�U��<����'L�#��,5�J�m�WkwRܪp~|�����}�#q��^Dr26MC/Q@v��J��_�+0�DqYx�P[ZU{�壂�Li��V�� �?�P��u�$=wJ���K�FCA�g�k���p���!��_P�D6�Tv^��&��^,'BE�����b�UwMo]�z�zR[o�f|@����;�dBN����س}}at¿e��c=�Lz�+�����U��:�����#=�.i�\��%h�H�R�vJS2b��% .S>xж>]���Br��؋�H~�ԏ�o=+�
"�92�bNB}���M�d �v#l�v�l<�"�$\,1�l�u�;�	`�g�}$?6B�.����o ���&��rr��E���%��rB`��ۇ�'K0��ı`��!O�TV����;�����7d �V�cO�x���� �o�r�s�5T�k9e����gb6Df�8��з�y�
52��5L�2�3�.>�!�c����3��}4���c���@��F�1�O9�hD�	p�8�'��)��>/�a[z������'�=�u�����ê�,�[X4ۦ5pN����JC�:D���	jwi�����}S;H?�o?G��@8��X��5��w�cEHv�o��M�}\��Q�wX�mm~&)���=|�Ū��sՋX�:������zL\����M��ߊ�և��=O=�3~{�ed��~���Ů������?�F�M���D��+���!s
�M#��o[�-` �G	�;���VU.�5�	����� s���<-'b.��5�-[U�$g�P�(�R���͔mNO��RB���$(RD^��F�x�xC�U��|s��r���-�����|[�x��kE���G���>�	�b��J{����Z�a�[�Ŧ�/(n=��p&*s���	��Y� 琜P����w�EAP���+������~B�B�I �e]H�ZJ\��,Gށv �g�;.��J&Z�G�e��&;�2�'�ߟ���ZӔ�'F�Y���'}�,h��Z�P��i���ͳQ��ͧ�N�*���NF��l=�u�SsF��ꌻ���+�N̷E����x�7�����DҶ׏i0����[���U{pX):�LJZډ4��ח��׬A��u��O~���-wuZZ�q#�ߴ�[�TtEZ���ii
A�㾯�.4���R7���� ]]�Eq ��#�I[�Y������-߬��踥��;��_�e�+����"R:J�2�V6�K��`�^��5�ed���I��s�dй�ٛ,����zo��UA(3h5�VZ'�/u���s��h<hw�̨��)��!�=z���Q8kI�%�=u��w�zw�Y8����y���hG����8�1�,@5�]�{P�8V���ڸ���oL�w\0�JMء�vg�/)�}P	�x�s��/�������jjʖU�ɉmg�C�x�v�z��t�ߌ4��)�Ǫ�YG�m|9���Vn�]R�諽�]b�>��qdy�H�N}60��لu�Mv�c�Qc�n�W�%7R8�@ T~(7<Z������E�u������w_M��˾�)Ӗb��f�cK�MW�#���³S��o���O���ԘT�w'�+z��_^���`�1řZzX��est�c$I `C����&�:ж��mm/g��C粤�ǧ��n�ʧX��(�w���k��K��|zR]�
a�Ŏa ���/؉�Pãࣔ������.�]���8̔��Ro&u��^��t�����-���q��dP!��D��v�=��̟��^�Ů�ћv�(FwK<ٗJ��T4#�}=��=��6�2V��;��\n>G��q�D�@ �����|]H�*� ��t���uI*<�/d���}k��ƀ�6�z�Z�,W���p��b�1L�d�2����R���U\�CfOc�V@�G�>���k: ��@�8���Z�ª�v�3�3U���Sӛ�&-�����e��/�,%~
�^�'KV��+���ZD�mVP�NSP�v���棵�h((�t�]b�z����gL�L�: /�4�~im��t�/kaL!m
Cn+(<���%���ׂ�:��u�>�6*�͵v�`���1�S�K�Թ�������2�H�0�`���1t�лp�@ `@h��x��$t����[����n�H��³�l�j�&��$*��+�_�Z����r�O�O�N�lٲeѳis���f=j�x���eK�r���(t�I�?���9�s�_��A�����醀�-�Kec�������Z���RXpa����t�\4�=�H ��\<��s��MR�&��<�H^m��R��&�j����P�`��v ��\ݖ��(E3}ڃFx ߽+}�ޡ�'������y:�@ ¿�wi������ŋCD��T��<eHu���o�O��.��,^lvy1Gj�f;�)��q^ăO,Hi*|���m���d�,��*	5��9X� t��U�	�å��<�e{��Ȃ��D��} �k��yW�'&;K��H�n�������e�&t�$<4��m�q,s<e�eE�+��t� ݄ʍQ���>���M��i4erO�+���f�HA��),o�'s*�:/r�u2��/��w��R��+�8��oPz�M��␀�"C/2���2�<
-��Y6lg�v?:qR6��-�f�� ����KsQĖ
�4.��^��I�Ԥ�(>c U��y.�&�W�i����3�;C,'BEa����dW/�n����<��J��^��\&M�!Lr�`�4��1��˃���3,���^^n^^�'�yZZ8M����eHwI3}�AK�$i�m�A�FX;N�pw�0�6����L�R�J�S�}=V(@�mٿ���ȸ�,��˔|H�D�mY>8qE���weö^\���:&�4�u$���YÞ�a\L��[�->��VHs�����#p�J��vL_IXP�Q��KQB2~UlPU,����+�h�½��!�Q��"cJ��qbn�\��J����(��w���n�WeT���?�P.m��^�z9Be$��;�x��K�ʾwV*c�p���S%Z�BIZX@���������y�d�<H圴R�h�}d�f�rW0�r2�]bO��i�� �P���r���F��a�BC#GDF.]0���y���Y޳��GFFFDDF��K:�[�0�($82"0d��4i'=�l��P/�@+�p��}���',��# 0_�0|�u��"Bi&��/(r�@ $ ��q�\�9��9N[�n>2"�z���ȠyRS�PAٗR�b��p�p%oө�m�޼{�?u%.�&5����,������Pz��"�����Ө1�u��f�S4�:#T"�	W�rq����w�J���=f͢zsf{y{xΞ�"Pz�Ṣ��	`�g#���,W_o~�������������3��f�3՝3����k��)ƭ�}��.����LO77���M��9�ߛ�$T>��}��*� ����'r"��bʧ%%5PyG�pX��T�P�Z:I��眞�:U$/��%�� Og� ����������%�J�ϫ�l�K����4b�71�m�>�Y?cm����b����@H��Ny7���@��O0��0kaa��kBMY�ZX���S陎��6�1H����|�:pm;ZgLC]���|L�K��������g���4����U�o>����Ӈ�y��YP�w��x;Ϥ^�Mg1b�Ј��e��t�IF��I����,]�[�k��z�o9�k�L��^3fxRo4����6.��S����΢8<��';O�g�9��a���o3�G�o�inn:�f��������(s���G'���c'��0)�,��(͆����3��'�������������ń	v���ָK�xS�1���H�4v�6�c0���MÑcm�l-&����������W^�D,
Z�t���"BC��B8p:���� ŊP9a�=���A,��Y����$?�լ ����
���kV�/X��(4,���0�%&���{��t�iwu}�.Z�9kɢ�E+���+�,[�&b�Ԕ@ *�KWD,\�2|�ʕ+�����ũ+���{zFD,[I�be���B�ɳ��"���g/	��;[��ea�2I�pN�q��+��s��haX��p[�~4a�Bf�-1�=H���_�p1@dX��v���+��\$�ߡ��q%�Wq*^��������~��O~?��@.��� 2�C��� ���Ss�"4�V h�@�7�%���l�m��P��7� �J� p�v8�*Y,g��w@�jO��E����o��/���?�[�aS����P�(P�w�ƳjR�ͩh^�=�O����_��̾bܒ:� �?>ǧ^�|���@ �^�k2��R���,IJ-�;+J�
��eu��(F0�5��'u�?�Zs|DJ�q����r	��[�����yך��^3
n,�Ƭ�8 #��������K��B�jm5ZO��X�	v�f�#���n],�Y���[��B��&�vs�Q}�����@g�n#;;��vv���Ǚ�;Ɯ�N�w�o1�fߔ0Ҡ��KŶ �z�ih����|��mjiaGu=���"*�Y��)��:����`����>/�{�'T`
إ�"j�Y�=��$t�=����(��y�mE��Q�ُ�XC�Q C묯�B�KVq�q]y�� n�G7+��9��c+w}�@��L��1{�� �ys�|�|g�S]��͞�?E dp�>+ 1��/ `�$]7�d�
 ْ��9�������fz͝9˅n�;3�O*���4�a�$��S�La��4[Oڔ��yr�	�@ �G��4�.���WEZ��ר1o9�/���	)����G�y��e����K�	pn�x�9;����H������	,�ݿd��i�u�>���qő��N �@ �e��ڣc��mgg�I���K�H�IΨk[��cm���8Y�t0��E<rϪI�i���e�d;����$tP�PY�u��TD�I�c�|8��=��/����/���v�s����ߤg�s����KR_�T%PzK	�����-���*�W�ŝ/y��\-���`F��'{��P��5�d�(��`Ѝ��2b�x!OZ��t�1�t����_m�շy{���z��F3�#�bG�g�45Ʀ�襧$�<�@���ar�q?a$/�k�s�BV0�[W��������r�
evV���VɉOz�����$q�oQ���O.ZH��*r,�Y���Zӑ��!�@ *
�2�U
�����/]�ϑvJ�ꄖR�J��[�q�ӕ����D�'-�Q?X�%�U�X�%yk�>L?��ɇ��r-�-�r>�ڴ������\��L���a��{�U����B*}L����7�s��G����O��Y���?y�|�
����ł�d�,e�{Br���2���*�;���?E�Cy�Jl@���T �?!xR+RS�PA�b��p��<DjYC�<�Y1�T��"2�$+�bD悛���'T�yW��9�B^'_���;��H{��J���o� **������=3���#i]2�"������v�-x��E�z$=�ؚ��z��ZK�K��r�4�W��?��].h���h�y 1m Ք��_�{�����
�c,x����H����\�@��+���׉:�w�;�W^��m��E��C���Z`�˻�!�Aj&@jJ 	�wB�\|�mY�kK��Et���G�:B+���!F��!g�H0,��-9�zm^,�q8��T�m1/���e����9�6#ז�'Tj�_F��pR1�FW���:a/Zr��Xj���oܥ\9z�zA'�O�8{/��%�@ ���T�ȉ��gN�6�bǙX��𼉮�m$g�C/���H��
.��7�MƩ�P�,�q���B�R��7�/�_�t����Ȟ@�p�:�@�Y'��L�ٲ�b8�.�����bd"�@���u��}/�������P9�>�D�9�B^W�FO%2�����>YX���EjJ �v���J����Z*eஒ���Mي@J����7#g��Ⱦ+�f$�c���@ �@ ���_���v�忡|#igm��^B�$h˞�As]m�`"������:��M9J/�;�����53�֦���c/����G-9�_��W�_�u�` ���W��Ӕ�����|)Fҗ<���R5�@��0R�&F��J�=\E�w^*����	-�*r��]J/���KV�T�d���OVE *�6���TREE  ����������������                              N�                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OCHK    @"     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �B             �F�OHDRM(                �              �           ?      @ 4 4�      ��     X                   �8             shuffle            deflate            �
        �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   6^�OCHK    e�     �       �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ,�             p��lOHDR (                j              j           ?      @ 4 4�      ��     t                   �8             shuffle            deflate            �        j           ������������������������                  d��           ,�            K�"OCHK    H     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �7             �w�OHDR'(         �       v       �       v           ?      @ 4 4�      �3     �                   �8             shuffle            deflate            �	     �   v           ������������������������4       _Netcdf4Dimid                          � K     c�nOCHK    PR     (      P        _Netcdf4Coordinates                                        	   �&�>                                                 x^m�]HduƟ�Ü�!]�X����2j�v5p�p�J�]Td~�2�E�x��L�!'��g�S3�Z\?NC��8Eetg�8��G��"��-ۛ�]<���x.�p����Ws��v��pӗ��M��jg�9_u&W�j������kZ���\����r�����>'�5�A�9�����#��j8�˰Jp�\Z7��}m_I�D[>�/�۰<�衙���7QFt0J��yѧ<���\vM��(eР��ݱ������7dH��D�ؖ%?5�y��R�z��I�l���j���?��NƗ�8X�|�4����*z�G��}�{���y+f��=+B��������c�X[h�� ;Ūd�\��b�?K�SĦC<�e,}^J�8��7�f
u[DT���]o�ɂ�^D�{aO-�0�"Ԝز��9���T��t���bѩ	#Hſ��8��n�E-Uu��!9�ƨ���Oq���-�o_�~���QE�����g�`����%男�#mJ�鵧��w�KP�T1!�*�^&n�bg��[|��s��a�~�u%���+���6�{ܚ��sU�tu^)�g	�!Bb�M䉯�_��;=���v�Ȫ9�Q��g2�������W���&w���X�$�W��WH/�3��P�-T�}3�;{Am�;��?Ae�Y;����!���wK�<e���u��ӣ~���T���@m�KNק������XP�/�w�0s?��O0�M�lO埁����Pp6�/>tg���?����B�z�TREE  ����������������.                               ��                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      TREE  ����������������J1                              �                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^�{X[���	��N(�4vH�8w6ۦu���(R�aU�����\��i�NiwԶ��4v��V<V�:��9���N8lzE���P�I��B	5��s�|�}R��w�w]���{�yΟ�y���ﾟ'�ߠ��2�E��\���oO�sg���2����2�����:�s.�k:��r.�k���n?��5}M�C�s�"��\���oO�e�EO����F��2����\���(�\��t�>��5����;o"@~��%�~B���΢����Y����$n�l:w.�k��4�j:s���d�����wI<��ϨZ��w��~I��>�McjWP�/��3/�q�I<s��l_=&�=����7��_- ���AL��.������&1q��R������G����6�;!��"[>����E���!��q�C�u�2��t���G���ܯ2v����U��?�y�͉���q\���ؠq�%6���_>�8J�W�D��A"u�����pjԶ����Ӱ���R�z�=t�WD�	�z�+
��Y x�������`;K:=�@S|)�2�~�œH
�x�=�M�}�_�lg����+��fA�R�7�Q�DH?���	ͷq��xu�����q!�M
8[>��i7�3�Z͆�ٷ%�
�lW{����r��G{��đ�6�!r�F\�R�7,��l������̢_�X�6m /&��I��6�B���Ӆf��ap�)�i�L;��Ԅ�İ�.��Q���!�\�l���U�޴kB%�³!�����ש���|:�{�I����k�/l y<�I�����{�n�Ykּ�Aֹ�H�g�޾����e�l�~�w��p�����k� }I*=�{͚Ģ�9ݻ��ٍ��p����7ވ�Q�ҕ��t��YK�Kw���^�{{ii���P�{�-����h$Rf���	D�����]׽�����L�v/ݽ}隥Ká�kj�u)�]���t�˧/��v]N�"w��Ȭ�{(*���\��Z��vZK&4�%`���LMw�s?Pi�,[%m���;�)���M���q�k�����4���+�c7_�q77���,tW������+�W�.4�_���[VV�K��| �з\UC�z<�?�E�j�B�~�{o���2�&��߈�ٹX(4}��!r�u`���e�؋�TO�n\�\�p��Gʯ������Fw�a8鮄�'�W����t����|wY����l9�@=�k{��-}��K�WW<]���r�Zϳ|�*Ca>��竀�CQ��aY՞EFO��rY���aw�Y��j7�j�����yk�)[�a�σ��x�^t��mkn���r+u�����iV�~��6N߻�W=�^��&�q.�Ez���[mD����a��W�#7��fOdQ�T��6�5oq��� ���o�֢��1����T8�����ܗ쯶l��^w�[w>{���O8�eOd����{J����g�Vрf�+{;3"p�V�'m�=�E���٣:M\؋j8��ņ�(��V��"�Ղj�o;��)m�B��ܙ�n���j�m�9���
��Z��fr>���7�8}��[9U���sk��r���;b����g����s�G�Zr�J[k����>~�6o��8~H�r|����Q%O��>�[�q��a����4�Z���H��м�[��g�]�.�նr���@�\n>����%��h�m��������F�{hǑ�#C�0^�Z���;�}�D�E~�Wts�;�����s�Dcի�ݿ����;��]ϖ�<V�������hn�,�v��������=Ͻ�������ߺ��I�=�������k�zoz�)�{v�s���e3w����W���xD���{; >��O���w����>xƿ��q�	/���e�o+��{>�����K�m{�H����s��7c��ue��zwx|��ڇy)���m��'�v�SՕ��/�0֕��5�yw���_k0���;�'���]���m��*+���M��O���^��]bs����10�p6���������_YVQ��F�J{?�!��!wӏ� Wj^�p�i[t���n�,ȹ�ol�E7�<02�yGm�g݊��k���mU��#˹҇�˸�²[�[�|��.������EM�?=9�^EӶ�Ǜ�+[˚���������y�2���[��#���\{5pz|��>���8�'�%�͟�;V�VY���W�=����m������q�[����?���|�C=յ;�m���ҏ��8/2�!u�cGeU�D�&�ɓ#{/������z�nl}�a[��
s����L{����6�}o�k������Ϝ۪��zZ��sT�ռ������p�s�"�6�\�ە��6���p�u��KW�y���Y9���h�����r���Z߼�x��v(W�v��0��77oy�!z"{Zm3+!��v�z럊����G?,*���KW��g�Ȇ�U�O&��1}�mه���\�Ǔ瞚4k����o��w|֍�������{�!>�����G~y�����㗜?]����g����W���4��}�����=�4���+<w�o�����};���T�wK����_���p�E�͎]�ݮ�O�^��ﴆ�����\�c����X����}wo_[��{|6��v����۞����dA�Zk����\����p�[�]�x�{��s7n���A܅}u�O�����Ͽ!V�O�{mcۺ�ʯ�c�����go��yt���l¯�ُ[��~
�_}7<Z|��_lu,��-��Վ����w�(��yP����ܶm�B7���v�7ۿ���z�۱�mv�M���
�����r��}7�9v��m۟k�>�����}�xʝ��y�.�
���U�-_W�8m�}��Ǜ��y����˗���7��6����r{�B����_���^��^ޘ��y����M���5{���T���N�Z|��=����{�Hs���F��߾n�wH?��D�ǻ�4�ͼ��=�;>:���w��K�L�y�5�8�ޙ��~����[�8���|���]�{W3Իɐjq�����;��hT3�����s��_]�V���:�.܊��Qlu�j�Ol�uGx�U��]��C�.D�x_��T_ͻj��q�����[]7<�������Sa�o���[����g���E�K�j��X	�K`����-^u�����k���Çٻ�t�GP:���[�l�?%��)���cwް����4<�B�c�`����6od�î�oU�{��C�Fhl���������*�����&���]�|{?4=
���͆�#-��4޵��{���#U!Gn)��l�rG�M�ߒ{gc�ڡⲍ?�T�44Un��]�V���~�+�F�u�WZ���V�v�X��-?�Ն;7�zZ+��'ZxL�`���������G��O<���_vtQ�߀�6���O�l�n<���|�'k�̛ރ�~�v|��I�q[��ݮ_�{]+D���N�����w�ֳ���~���<ܸ�4��u���l�P>����?m�z�?�O��x�e��3��4?�����Kv��A��ssR�n�4pGVV8�����<*nBWmn������qfn󛮾q�"��.k9G-ga�Z�ҹ�ڍ��as5~�:<柁Q�z������y���6=�E�-�#})w��j�ߐ���Ch��"�I����}��ekؐ��C�f������@'F�"���hୡ!���3�r�Oy�V[���6��%��w����C^�LG>�A���ۇ�O��S9���ex}(F��^�K�xVM�Ɔ~�s9Nץ��m��K�:���NAa���p���~`��6�;�N��6�}�]mo9�.�N׵���0B�9��
����	@�SN���*���!��f��sᇍ�Y^�n����av���x+>�.R̄�$�D�Źl�s��"(o�@��:P��^�Q^p�Tݖ6Q���)(�R,]]ʕ�%M���� �����;���▰� �HE��[ ������P'�@0cX�0}̂_gsz-%�d���p�!U�>N�U8P��J�=�3�0s�� �TKշ ��p�z�V`4�}��eQ�x��049���_�;w8���cx�#^���¢S��S�~���9������@v�O͕�	��c��K�6o����1S�u%x����A ��.|0���	�+a��e�qy_qV�Q^�9t~F4l��kqc����s��:EP��Be�"�6���J ��&�⬳(on�7������z�w1�]uJ]U�2Fp�N�8�:懃|�S	(�\�Ex�F��.��j�3��Ѷ�n&0+(�MP���q�C;�0��桮@�>~t��=�ev~RοYjy=�9��ޒ@` ��Ǽu?T��mYA e�'�`�˔|�m���'l^�<}(0J	z��oQ\�(P0����si@P���������P���`�z�rTQ\Ne�S�=�\;���-@s�FKJ>���z��b�(X���40��
+ε-g�	� ���𹒶�^@�����+-�p�8�h�sw�_�<b�k�i
FI8䃅8
�Z��
��f���8(G�ۏ)����襅:���!+C�������k��؆����xc2�tÒ�P��A��7`�����A�tjf�ٯE/���n�A������NA�<��M��� QZ�0�]L� =�i�TM ڝL���;c8/M��0sр�~:�/�*��h��֭��p@`b4���؞�@��Jat�p��u�����N_�,�X����b1�Fq6���P`靫�ӡC`�&���U�n�y��4F���do_�-���g�� �Cх;�/ch#۝BI��u�ϝ�K��I�R�d2�ө4�0U�iJ�\K�7�'Y�;3�R�TڮU��ۅV�K��1c�sB�ә���t�}�w�G�JI�A�lY��F�[We9�7h`G�ި��\��;�B���1�<FY�+q*��F�
c��� ��P�װ���P)��w;
�v9�A�B/���b:,uU�S��KޯC�ca��k�t^������T�S��N/Vyݥ�@����x�Ki;��Y6,V��|#�O�;W�!�0�6�ن�3�!���b72���^] �[,duU
�d��]NaF_�{G[ �Z�U�ub#ڭ8����*0��1ޒ������(�� �cQ�n��8q���>�z�&E��v��T>���ɫ`uZ�N%pkW� ��`��+�ׁs�8�pU��9�0e:,��
V��*�Q�������s�1an�p�3�v8�kEx����a�u)5d��(R����M��k$�?��-mN��V!��҆�"��)�4��E�e�f)��Q�1�%X�<�C�85|�b�C���-P,��W�9���_h�Ɛ�{#3�"�W�q��
��5��`�)� ��.%OP�٭������6w��|�+�AH��2�roL�=�U�d�^E�T�㌍� }q��x;���!';�T.:L���9�A:'$+�dweW`��q K� ,��rS��I��),�\�&�[�e�[,�sT�]M�٨5����b�?�'&X�� 6�p؂[�����ԑ��%m��rf��0�^�׃Z-N*$|�f#T�Л��]fn����:�6:�P�1fP,m���&4Bx"�`�z����=xP�t9��$0U�+��)�L׷/�M��I�b��Z��_�e��:eMU���O�0x�b��NKxWp��̢xZ!0�U��n`,��{��za\���e�x#B3�����u��GT��������L;K��`0֥87wWu-��.N����;Zf�.�0����u/=�g�DXi�<߃G,Ӻ>Y̔�n���~_��²����+)y�J�������ôyD`�X��#X�o��
Β�kƧAXQJ�%���*�"a�N:�À`��P
�%���Xp�h�	oP�:�ҩx��St��i��l�m��8^K�8Fjn�[� 8�t:ufA�ԕ�wsE��
�u>^`����Y���+����e�M�H'&�L��:|��Y�du.^V�Q�QK���Mx�+���I��P�SԂ���&�s���2<(��	��oXs�(yRʦ�����<������J.�+bmQ]y}�KTW��T�+��j�k�]HII��)g�+���*�c��\j�5�}�
�.����\w���}�5�\ʫ���J1Y].��ˇ����d���PJ^��rM����f�&��b��*N̳�״�����t��UQ�$)E��H.I�Wٖ�y�{��&�ʪR��
��
�t��U`R%�_�ת���V]�
��3��UmEJ���?�)w�Nئ��a�¤� .���o`Mv�d�V���G��T+[�2�?�Jp��T�T�ZD��?�XYI�?�|X7YWI����MX�� Uإ�M�Ak��ZU����Py;��}.�]�7�Д�vU��ݬT��W*n��5�}�d��_�@�NMmش����&k�ZV�N�T�y7�w��4����^�
+jZoU{$+�A{`\Jl��zm��5Qk��(O����
��K���*�֛�rI]Q`5mBj��qVb{�ұ���Fo��"��X3\ъ^��8��T\�'�
�R�+������Ҫg��ҮM�n�>��T��s�� 
G�޵�Tk����kT�BRM���&�㡅�O��V�Y�w'�my����v�f�e-����)�
��L�a�W�j<)A�i��֠��ȣ5�*��;�`�����"�U���%	W"
$"T�5Ku=T%6�<�^�\��K���^u0�N�ϗ$+9XU]�Oz,��wN
�H)/����ծuv��ŴV�K��jHSC|�Ub�[�l�8� ?8��YR��-�
g�۪T/���W��i��4�'�-�iFI-�7�(<og9U�\��X;��4�]�OR���D/�.�ƒjJ0������Boe'R0L�v���˅���͵�M���^�U���k��aյ��YU�Na53j�J���4+o�ɞ���A���T�A���ը�VPTu>��\�]L�
-v��S��κllqv��*��}C�kq��WJM���+R�i4[�Y���5��h~��~>�����^\�n*W�?�j�����ͫOaS�F��*⃝W����wՠT_/���9�	Jl(1���[�p]����<�a�K�i�����L)8tx�H�Cʳڂ�/ah���լ����b?����MR1�Z�Yf�5Wqp��Rn��),YZϮ��R
+�<&��s�7ȯ�׮��P��?���[~},���U��!H��Y��z��/��0��D�Qj�-~��:�I��z])�z��d데R���*��Z04��z�ū�dg�R���ǎ�C�Z�^J҉n���_/a=��E��X�
S��6�����d�X#{�5a��>�,��0��TH5���^�;�C��[ߍ�S�˥���`Z��5�o��M��k�b׺��ѽ����vu��[,�X���q�n}1�5�Ih�1b���a#zZb�`�HET&v��)��E�:	��k�n�=��!��:,*x���7���-[�FRoĺ�b�YT��MV��e����
R\p?��|���},�v��"L]�v{�I�ʐXl�XU�_��g�Uv�Z��zۀv��kI}����5���) �$}D ���B�H6*$��OkXvp��tY(MQWb�^��O&�:Jv=��`�֓<d����S��a�C��K����W尓�j���`5���׺��qQ�I�p���JZC��}�!�đJ�%�E]�ii� <�F�`�ڢ5tg�AH��*b�� =�O�'-[E�A�I��W���З�'-	]�J���Gp���#�p��a�k`�A��	ɒ���V�a�=�%� v�-.����%�NQ��P��qýwc�]e7I���-k��Ƕ FS�P˷-�[`S�Z�l�X,�)ԑ���"5��a�ւ�y�ݔ�Z�,��f��7�r�U�s64����>Tp��@�� ��F��^���Bm1u�?Ԃ{�K�$�U�r�V��j����e¾IֲT���U,f�G�����@�����R_�g��\�ä�n��
��9���%��V��?�]�^���8���X�.j��֋+'���2�2�Eh�9R�7���V<� ]z"�*S�����hg�>�.��P!�oWx0��5�~A�-�S��\s[��ޔ֢��DXGͱ9��)3*����ۂ�G&Y��z�U=Iu{#��b��4�X���MW�)��p�.�`��c�Vوq�3�*IS�i�3�j{!�8���0 �h��R�k�M��j��M)����d�˓l�Iɓ�-�?�D
^�{h/��1��1R��z���jh���{M�ly��8��JQ%���}�kT}v�-�U���
Ԙ�����F]K'&����^|��^<��i��)^�@{��z�B��q�U]��yx��.�0�6v��B2<�P��^־oX�aI2��r�ede��!�D}4.�b�q��49nH��9��y(9)�&�Y��)٩���Y�'2�Cz�@^<.g��3����H<���?iJ��9�h4$������qŘ�	I��1��J���cH��㓍��X<n����cz�gF8��Qup`�Н2yJ_h<I7�ca\��C�@4
e��N;�>J��'�1{2c0�������������Ԍ��2�e]|J̠7ƌ�T�����=�f�DFR�q3�A6bX��<u �?�~z�DڔQ�aqjL(�g��p[x�~��d�f�_7�}2����q�?:E�`�f��o��7-�x���䨿O1���͐>�l�!%��$S6 �M����� 9��6���0g�'�$EG��줬���L��a�$�C����i�)�)ƜX�g���c��X�Tt��x�d��=Θ�1����A��I ir��q��NMe/̊��l��5�1��E�0
��q�L��1���7�iL2��\8��a:3�� �W}�ҫ&b�Ͷ�Kpŉ�Y�~����F�&�`L�+���d��K,�'�Y
_�Ťؒ!9$�Y�<���Y�31+�e������`�9���^t���tո�[�ӾQ��%g����c�'O�MMN���Of�*��0&�?�rQ�Ѩ7.�E�,�2\~������,)\b�?Q�㠋����}�?�}>��=���$�c���>�;����~�[�P��XN��ߺu������_����ݾO�d��?�r"� 5��4��]���<(v�^ҟ��c� ��1�~]�����лݻ����~�nq ���?�����KV�N���r����A|���/�����°��>�Nv,�3��"��ė�7�}����Q�?�a Y�c��>Sc��z���Ѝg!Z�FClr�Q/�Q�N�������23���3!�񣧠O��-��Re�3����l�&�����Ԕ���3���q ��r밚%O?)�����9.���t�>����S������Q�<��$�2��|�9���I}jZf��S�Ա�i��TH�˖�F�X�^oN�3����ӧN�����e�,s��OI�c촜4%	dq,uؐ�:��?6��'��uI��B���4yL�8-*i����=��N��}��'��'����Řr0�����ui�G�C{tiY�>�_=ΌM~�'�2�M�>�[K�����a��;>�&N��ԉQ�.>���8�4:61��z9����7�O�%�/ͨ���g�l�13����T���y���M6$��	�XR��$}8��"��I�с(��XL�?I�;�ޝ�Y/F!I7͙>%�i��`:�����	fet��~yr�@��~��i���h�y����ϘSI�U&�]4i �3�:y(g�����d�	M��3�F�����|*SIN�J�'���I:5��0�pF��x\�����9� �Y��)2���xl�8 ;n4��r��ՠ�ŒN@|"Sԟ�ۓ�1�<O�
c�M���0��щtP��|L�ޤ��C�Fe9>���ON�M�<���qx9�}��}�Łn���Ϳ���_�O����~��'�?8�G�)q�'��ӳ!5=��7|������@w7$'M�+�I�O�Fs�r<.vw��/��^�c\�7�b"LJ��&���7��N��>QF)2�e����~C��a@���!��}'ܷO�>����Sp��'���������ɒ�}�x�^�������/6/�+��X,�ץ��Ӣ�a�^�?��ſ�¢8�3������gt
@u���I��1}LN�@OǶ'����q����u������D�SOW݃!]($���|)�n�"OD1"��B��"va!�>�Q�8x�M�O�ſ��S2��!�/0p(��>a��3]��އbr6���Oİ���}�� �"�{�'d�G�ae�dT~`ʩh橣SOM0��I�w�f3��"���Qyb��a�OG/��eM�xf��K�y�/.ҟ��d�lN>Ņ� �7>E�HON��\?-sq�w'b�C�i9hԔ�� �M�0g����OΉ�_7�b0����ebIc�ō ����,y2��06|�/-mr3=�80�>��)gc;�7�N��tSf�~"7;˜���,=��k#����yzʿ�9����7���^���EE�-ȣ�E�'���30%4=�p��3�Ⱦ,�<���]�_��Z�������_� k�g8C���٣щ䉌���� �[���3W�Az��/��ŋ&F�`�͸lɯ�a�����hL��z񰌍-�g����4���+^z��1�<��+���8�zŒ�/.:f0�M���dSa���c��:]�y�i�rΌ�9ϟ?}� �|�8��9��_ OL�,��0�C:E�:%N�,���� �q9Հ��G�?��u����?����i�s<'��憀Q��r��`�I��s���:��g`<��;���<f䚳C�Ag�aC<���I����Q��9c�9+YJ��l�������ϖu�,YO	��#�����6��Ӌ0��&sn�i�ij��X|4&�Ubj�����X�;ۘ�v��Y�I�<}`�7���5������������_:��������::��
��p�&viӵu�E8���A=vB!R�6HSpu�6����H��PJ�P!NH,� )��B�L�x��rd�]4�$Э�<����w��-,�9PH��Jc�͠�G& �R8�F�bRJ��$�A+�uTZ@h�
;hA��K�"ϊ�pjc��4]B6�� �ٌ�;���G�7��~)%(.F�'
j��ha`;)$M��Gs����y�"����r�
Z��`�:�M#EgFI��C��)u�Z�i(a"	 8�Kt%�43Iab�6�F
���_.��@�"�E��Nm"	 �d�F��mz��|�U�+y�y m���B��3	I@�Gh���FW�73�9E���f%B��d᱈��dr_#E[SuF0NI`SSKP�8xE��(�_*�/4u�Z�� Uj5VHn�҄l��h��!�S+-�	P�5��@z�Pi���Hτ�<C%�$�
\����6�PI�@�%�jSYv�(�TB��H�Z@%H���R�O�&�N���0N!]Z2)��T���N���̥l�U��Je>�|M�J���M�Q�	��%��h�"ʼ#ZDeBl���'Di�o(�­�.����頊� �	��P��j��|�0A2�k���A �Ix�J-#3�.�GKM�O$����Q��tP��Kď$ݵN$��������H�Ly=�+��%�A*��\M;�|נEE���,�jq��Q�"���f��|��L�vj��֭���I�O3��%�h���4&����gP:�0�
o'5N*a8�/��詐�� %�X[O;K��"��j�|�)ZIj"��qF)5���CFj�
I�fbSD�AK����@4�4�5�C��$�=hS!�	���Ч���-�b5h@��J2�V�GK���Ц�G�M�FtӐ����Z�E(#�h=�.J���H�O[�b�
��u�t��֖��5����B,n�T�Z�5��B�!�b"���5��������F̓*KC1�&����(���������"�I\�2�����mP�!���kZ���ʭ�|�A�Qh(�$W+RjC)������r��@X�
��hB)���uM�f3y��(�4�
�H����-�(��D{Ӧh9�� YD�M�Dx/?!;��vh����.��9ZB�HSb�&3�G�#A��S�m �x$�	�	d/yLKI�v0"�g~40n�٬3ѐ4�u��4��M:~tP-'�v�j��RDm"a�F9x�AJ寓�Y�$�O#�)�|#|�
-U)ZN�(�t����_6)P�TREE  �����������������L                              PX                    �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      BTLF l        �  + m        �  $ n        "  $ o        F  8 p        ~  7 q        �  3 r          # s        8  ' t        _  2 u        �  > v        �  ) w        �   x        �   y        	   z        3	  > {        q	  ) |        �	    }        �	   ~        �  %         
  + �        �	   �        �  & �        �  # �        �    �        �  ! E�A                                                                                                                                OCHK    ��     X       P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   �,�lOCHK    _�     `      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �"            �T            �p            �r            ���]OHDRu8                f       �              f       �           ?      @ 4 4�      �     X                   �8             shuffle            deflate            <�        f   �           ������������������������T        _Netcdf4Coordinates                                              P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               p       DIMENSION_LIST                                    ��     �      ��     �      ��     �   ��2�OHDR;(                j              j           ?      @ 4 4�      �     �                   �8             shuffle            deflate            =�        j           ������������������������H    
   is_result              @                                Z-A:  x^��C� ��EѴmWڶm۶m۶m۶m�v%�o�!ܨ5����UP�Z�<Sy�k#^(��]�s�s;��Ǖ�S���ߴD���@_˓��?�
Mc75�\���X��S���]��Ԟ��u�{u����ߙ�%�T;wN7�x`���D�sIuf�T=^�.I_���9����nu��+`I�6S�EY$c_�ap�O�[�iZ�u�w�I}lO�?9!4u��WzT���Jp�c����3RhhfX}��W�t�A@ւ����?�y��;S�l��8���a�R�vx0��
�� %���|̾���8;��
�Pr%���}V�X�]�����H�����5�}[i�;F��2��1�J_y<~#�h;J�w[ZE2��~�t�X����qߘ-;5[q�R��] ����"����3�������s��Dm՗?�=��Y8�P�^ӊ&�e*Iy#Z�0��f�-���収�k�3��C�_��h�L�f�f_b���2j��m�x��j�
D>j�1�O�sRF��|y�M6���]F�xae%aeW�x��l�W��S'27ݯ��͘]�v
"����tJ�l���}8_���C�+}�g��uG{=MB����t�l�M(������Ѱ��r�S��������@�J�J��x��`Ӂ������
�c}�CC��Ԓp�S�~M�s߸�K�tkQ}�Jk�T�����0�kEm7ҎirX�q����1�2�\`�9m6q�xrP�7���X2rL�.e�m�D��j�<r Wcni�s�<����{�@}u��H�Ώ9�k�R�d��Z��a����/�#7]��QO��B�BON�@9!r��ї��;E��n��~�/$�C
���m�w�WU������� G ��SN�T�v�����Z���bz<g�ت��Jg������������j�ۣ�DmeW�P �H^Y$R9ۏ�`�?,^��H0��� BP��3�?��v��v�8�����_X ��j �Y`�ރ��˫������9����d���ʵ���_l��� ��'�_�m�^��_�rt��g���Ŋ؝��- ��ڄ ;.��-�;-�ԉ�đ��Ę���S�����,�H��V�҇���X������롢�Ό�䂍����=��������]�F�X%=ܓ��$�ڟ?E�o���VX����[;�XQ����K��&�G9Φ�p�9���Ƒ�|'|���ɬ���p��X�T���_���<�ߍuq5�������������%��ᖾ����(�71�p��B��H�d�qhL?�'?`��;Y�ԅ����af=M��ʎ|H�0+WKڡA���0�I���IC����󞇡*x�+�	5eA�=$9'��1�I(�0�w����+�q�M��ǅ���5����x?�\FW�hc�� ;dϓ�"� �t��FBM.ߛS�]�!�w�U��G��/N�7s��N��Wƙ��{<���?��$�<������5s娞���]V��`��:���G�+�n.6şcVI� ����,k8���Fh����Z{��s=�<?���gM�Ҍp���`vE��.4��W^���ZK�ȤT�ËB�+��*�`��X$f·�źMk$��Y��h� q�񹾮P��M�V�H�"*MSq���b�Ե���F�g�C�s7y%@�@�����]1ube\�*�V]��{m�o��gB����F�u�u���S����\�V w�ɢ�x�[��� �er�#v[m�d�]F_:U:Z�&:j����{#?&�kK�L�C`���H����T|�w�j����5���L�4�h�wRn�Ǡ���x��0�ݾ�ɉ�Ӑ�f��k5S�g�ܤξB���c?���d�BT��G�Ѻ7�z�a �д�P�b�E�e{G��O���7U���� ]aI4M����^�bҋ��]��x���	ν�L�k�o�������y��oL�y�w{��d��]�ygmV�e����E��O��r���ǅ�Sv��WoR��S1��)����6߇�������a�AX�[z��� ��9����L]�K���1��7�yB]2����Gd���:)i6{<<�zzgM��&�9�������\���M<�M��\����K[��z����RX�[z�!�Ig����{BX?w�Q�ã�u�N4=��������{��3�$Z�FF�a�F�����-�C�ь������|�������jĞ:����%��Ԕ�<힎�ڝ�O\���-ݰ��!��ef@��������r�׹���V���_3���O|��*`ڕ����ŘQ���,�o�%�\�����������r��|���K`1���o��P���å����P���q�h,?�څ��-���w���7E}i��?��*RM=�-A�T��v:���E	������:ܟ�5�k�R�����"q��ۂX���O{����87���	�1FߠU�LDG�������ȋ��o��l�������P��]Ȏ�]�-����.�澪��1�È���?�ms��1�J���M�x�ҵ9�ƣ�Y�7{u���W7�U���Ž�F���3��cY�,�ٽ�]&���s}�;�G�t��"`���p��j�;r*�F�z4�����"��C�!���rS��,�b TY�i�tO��Sk����Y׿w�l޹���/v7>�$.�ħ�	��j1<�4+VFE�Х�4��wj��(ȫFO�b�[�|��##��*;��{��`�[:��s���ȏb���^�rpF�ˋj��7N�� ���u�G���Ac78�U/�Zѐ';ގ�8�W���W���^�I�[N,p%��'�B���]�]��@�y0A~�����%�,��O4w����uL{�����	��y��U�C#}�!�Zو�V�~TuEg�㶠���HS�ɴWϲ�d#{5YWѮ�����	�#�SH��~�N�u+���F���<��[t먾�'�^O�s|�w��,(��:!�^t������;��������������4x��v^a3�f��`����&�ncB����>g��<"B�Y����T�/7�K�����uA0�#
Ƈ�n�%��^�TVljG����7�q��;�kk��ԑ�5���f�^7L�qi����-�.��OX�@����D�U��힇s8�.Fnf��F�����aW�NwR*���7�|��y�s(�0�M�����G�{2�m��o�Y�˝���A��e���֩��a�n�����|�\�m�t`��c�/|�q��Ȕ��l�JS?�)�� �u5�B-�^,�]�~�dgS�a����_3��wIŏE5��/R>R�F��g𤋮N�b�/T�h� ��o\!���)B�� �l�=���v�EE��x��}�V�7*'����^),�u�%�j)kI�v?����ٶ��Ē���Q@[9s�l�!V��.�g��0G�Y�^��vqX�����p:���7VY�-7�(���r"�0��h@��8�<���d�e���3ʟ�,a�eH@�H :�l�Z 5��%lڣ���԰SbF��r'�������3��Z�u� wk���rc-�M��KS�Kau�C�<p_)֐V��:�孜�2�"��WTb��:"�7�7��s˘G�ٚ4�EcBK�o2ѼD3ukB�#��áhP��7�"HY-Tؗ���5�`=	��h4���6�� �`[���yO�|��6w5^	�^�	-yd����y��_�ЦB��'x�3$����M��#0E'�W�&�qĄ0;,O�t{@��1�p�e�M���W	��w���M�yl����_�q��(��5O�!�cp���yt�I��e �Ӆ����P�Ǣ+�ѿ�X�K�� ����V��?�p�x�H���n1��hz�8�xIR�t�w	�9��m��aИ�G���V[�$\�8����s�B���Tq���j�g6���l���z.��|�7�B_��4�Z
����`Z��0��3���j�q��I-�a��2r�l�� c(�|/�'G�8��V��b�W�>�}""� eCH|1��ו�"֐4d��h�.��:������e��*���,� ������Vi�O����4��U��dq{Y������C�+:L+�Y�;i���	�)�%a�HY��+XmJ����W��$x�������0�4~��'~����|#�`��B�7��Ӹۃھ��"��6��ډ>ud� ��U&ċ�\����%���K�[�&�^?glS��`��[��K�oP�3r��{�3K{�(VoT�8�;'�A�5����;� u�0�k��w��6�l�UQ��Br��|*5�,�6���2�z8���vR�̩sC#_�!D�y�Yp�/��N�۽ßLA\_I���y�����i��5ɤcS5bS�D(Q~5�&j,��:�
C�jv����n��b���-���b\����0�x��4�)�e�f?��DM�H�;������������ꩾ�D�d���̘���΢ޔ�.�b.����0��Hew�ː,,FqƃLqo��˟K����8T�Gۢ:f������$���f���1�ǈ�綍�K�I��K��.= ;��k$Npl�1�>�ԝ�#P�
e��.�2��ˌ�%ģ���������z[�k���xa��˒��P\�b֫��]�����s3r���C8�6�v I�~��;  ��������#�c��+񷿓��X� �5����hc�4�\`E���ǆ�pf�!/�t�<��8��������qe��l�n%��T�id�s���6ļ�3���[�Sv(�F*LƐ�h�{v����D��o��'�t�hw���������o"!��:���K��@��O�mwM7�b�v̪���,%�F�#1�'�0�����OԊ^��X^C@����CBWQ*�~� k9'V�e<v��Кq�y���Y��6�������������q�������ڋ�\�������WJ�A����`~��������p!��b��j{I�|պ2���G"�8�) ��v�dj���I	6cQ�lc���gVkn�f� D�T��-Fshf�Vי��	����g��H��,x�kU4)�=�#!���F���SN��dܖ�F:qý8��̓�ܸ\�$���F��[Zۘ��-���]���gG~5��T��(�bL-�!Ma]��ya�8E璌C�Y�A'7:�x�Q���Ṅc�6�T&��St&6���2��G�l�k����o'r��x �&�JPst� ���0���h�R|#WY��Vb�8��c��{�?WJc��u"ΒyG�ug�Ҽ �v�HJ����8���C��W�oS�5�(Z�������%��ץ&�*����J�n�����ԠM�d��+�C����E�G�A�:�	�#'�o��gؗ1���X}��L�������~W��v�H�$�9�����o��ų�kS���M�E϶��i��G����.�*���QA�O��D�g���+��]>6ac*��l�)X����KA�.�᫈���WWՊu�l� g~���e�D~@���#~��qb� M�wߗ�����e7�e#��E�������`�W�0��*&���W����Ge�ê�u�)sLy�xRQ��MdY�f����Z��E�נK���$�0lק��]���ѝ<�F�Ç�als�bTC����m�"��������n]�>�#��מxߞBN�Z���(hYf��ֈ
��¨��fKߪ���d�Z� �����E�4V�Qg���� [����=��,�sO�!��E@|ޞ�9JD� D��~�E�!�<\%	�KG�`DQ>��Pb�2�=�R�Se1���0(3�P�4�[�{����X)x6��{��'����ʹUj�r6��'�ty֓`�DA==rJ1d.4T �պ����?��wXr�}��q�E��$c�Xr����Z�����M*�6����(���Y��~��a}z�O�& K��漮u�U�������N������
Z��ɵ��,�m!�|��U,�h6��Ҿ���|�!�^ 73��6S@��T�S�u���E�)��%���$��с5�9=��-��{��1S&x\���"f�/"���xf|��<����=ȧ/=��c.^�m>�����p�3[��e}�Ҝ1�7�$M���D�8dY�2n�&����n��j���S@�֐�lhT�U�1�r$b.<$� DwtM���"��+7���t�v�$�ڇP2:��&]I:svFh\�װ�T_�(�H�D��j����y��m�E��+f�~r�	R�^ƄrZ$�08�+ΈF�$�;�7C��T�usT�.�/�4���~PȻYyI	�U%�|GK���]��l���ob��% ���t�QQ��g̺������Y�>9$��N]�v�pI;L���"�D���~x�iM^H�d��������T�������"��&v���.̙�!�N j�J�}���J�M_��K��dw��S{�5�#m�X:�PZLb�4���8NM�к��/���]��)x�4����r�/��:b�-P^�;���z�5��&�3�B5������Q����e��Hyڬ���DP�K.C����_1k�ס�hvV��m��j��z(�?>�+P77pZx~ib,Tً��d�3�����~�X�d*w�)�����8�X^SII����Q�I�m`�%S��*�V�~DJUN���0n�1�W���A��P�� �������"J�qm?� ǧ[�j��r{x�}�3��;��ĺ�O��;B͢�z�g׫�-��#*�,��iR��!>�` �C�2�k��k�i*ѿ%��@��_64Y�Y�IV[74�!	fG�,�f��r�h���֫-(�S���l�M���D"�י/�f��t>{�p<�@Z��R��%f���揪y��M�#�x�2繻��@����Q�nh6���&=q��=1/K~0=b��bי��3*�9a�B�LdM�zFR���� ^Q�=�F]��i�4-.S�.��Ɵ����J�տkF��䟣�2�S$���-��@#������4.�HbK�L�6��i9���!�:�>��]���!��v��1e?.䋂�tca|�+��tLa���G�.F�r��A"�Y C�+(T��}1��*)C�B����G%�x��P�Sy�*��Ä�1x(|yw/�4��5}"J���k���j�6�F�2x�h���L�4��9�e���'�����4<ʢ&���u��u��ʷ�Jn��讍uT���%���S����ШV����ﭑ�w5������v���	ONUM�-Ha�Ch�5����Ŋ�b ^Y�_dqٟc��J��A?�䎪�M�|�'}��z����k���Ђ^Y��.h�iu�;d�s0��W8�v%�̍L铙U���������)��t�͉����P�qx������.DlhZ���<*9�e�{e�l��{N3��36���c2�t|@h݊�[4�dN�7�Q�����$O�r1�<7c�Z7�Nx��k�����c�{ǚ�uܦ�Y�+�$L�u0C�ϣ�]��]�?K�v�\���QI�e6�?��@Agh�A��6�/�59��_Wy�fy�G����ղ���NQ[o+��mO)ǭ�Tt<����]�����M8|bb���<ܩ��9�8�;��$��Ѣ��>uذ��F��sQ9����ԏe�t���指c4�x
N��.�us�U%ؾP�HtP\���E� V�!(&�D�irgbZл�0���g�O�h�_����Ѿ���f! �ވU�؎��R�+���9L���⎨��­�}���p����Hq�%%�`良�Y��LQ!��5i�s'^vNMX��%�E�^;�����#L�|c�m1���l��BSZ������q��Y+%&]?l�KY?G���]��d������`Cig��I�w?x�H%�}�pb,f2�q��0�Ţ�����y��u�(또Ӈi�4�iY,Xc��!��U�\]�t;�c'�2s�i��3�>�(N�?��v�KAd3����y�y�l{��ԌK��L�M)W�|.W�Rm�ѩc�zS�.=
{�h�������F/�.3R��]��_��D�%^:�+y��6�l'T�y�<�}��4�$�m�Gz���n_�=܀S��^�����]��}��{�d�%���j�d �`�����loOLăj�%};P�֐�S9�Tx����x��e�u��b/G����Ns���Y��q�@��z�$\f1�h�� �׳rGn���m�7��]C�m�b�lZ �R��m��lIՕ����"]�`�[L*������#u�(Wza߃jh#qo�<��z���۸��G�$�R�xӞD3�[L��L���[�}҆�M�`������������$���aᷔ�����ܠc�p�I�G��&�ښ��#��ع.�-�tz��I��wl6n��c�
.�4�@������O� ��mn���F��7<��(7�-�,<)_.Y,!�)�J/>���	��c�n�
��G[�dTm�7�֧f��z�I�Y1,�3�PX~]���B>Ex6�Q���܅'����G:�������l��ڒ>"��7�h��R�V"�<l�W�]_������*�n�I��>x-<�;��<@rWsv=���6`0�vF@���%s����(�xpd/6ӁP���F�z�~?�$uA�L��$����=b�s�֞]�p�EDC��z���B���M�A�S,FH��&LB>L��E���p�}G�LL�X�yk�c�':r�'fѣ1ac���%B06�����U��?�C�SU�]��R�@IN��?ۻ�N���ᓤv�Z8+{s����w?��aDV
�U��	�ݐko$���?����H���`AbPЪQk���z"�x,�B�s讖����9YNX��o����ó�0{9����hQH��f�)-L�+��-*���׻��}���\�5_��	��ʔ��YS�:f�w>S�l�{/w$�g��L��=���7��!���&�)U!�����VغQ�����:���B��e0�ާs�W�[|�R4�	���Vk�䆬�����g���fN�0��X*	?Py�,�]ԨW.�Q	K�D��������u����$N�Qn0$mjC���]#sW�iI|8�Èd�?+�i����C2V*��0��s/#Av����X��$��{�z�-X�d�P�"N��ȘV��'މ"d�k�nF�̩�pz�{�׃\����yDt�J۽+老��c����?E��B�D#�R�0��G���*|��QBj�\Z�P����;����M�����O���lC�	�@���
c�,'P�U�Axv�<2�Ϗe��ɨ����4MQ��I��F@��򊺛C���,c�L!m����[H���0��#)c�k/�p�\ܣ�mAN���!0?��扰 ���h�
{��#�0������ X~��� �FB���/c��.�Cr�"�]�r�f��ou/�M���:����
U#ѱ=r�6�%�7l7"�bjc�ap��4�y:�,>��o��M\�wV:(�#�����邫+;�!Q8\f,�^CZ��;^{�GzL/�}��a@����:nlv	�"�����#~��δ���%#��&���7lX�;T�u���G�Ѝ�cBgvC�F]�'�}��	An�D	��3��FEv����G�׻ԛA�
�Fh����l7ޚ�\������I������Ao�b�~Mq�X�%N�k�f{���L}�%�KӐ�k�d�t�"�$(9���"^,�y��O�A-����%c|�
�@���@�j�.�|�����L�'k�B��=��7Sc]����r����.b�Ý�3{]�vt|��逗����e�0��^*�+�/z��S,(=&~d��Q|!�\�s+��$�����I����'�@���;��Z�M��W�[�8��㢯��i�^�*�
*B�yk?(t-�2�+*Kҋ�M&�ǒ�[�����Y���{X��Ў��JƹF�d���6�������$��0���`�|8;%��M���lG
�%nZn}�z琴�"�|5w�)�[`�6_��/�g.3:j%2����X� p������Z9�%K������ƌ	��w��!�AZ^�~�I��<�}������g؋:\���0l�<
�+d�W��	JT���u�'�0����������2��[�ҷ�B�^��G���g�Oy�)��`WLH��pn�)Q./j:���'�\[�2MY�Ї�4T�ɴ&`���qS�L#`@�[̬OL�y�E���<���S�$��h���ߵ�J��������E�y_�x��A�0���P������)����������aw���l�;�o�NɆ��W$R��q߱B#�pi���k�}���9|� Gz{�7�%#���ٓE�5�I?���4�t��%��o}��xc�cb�%��n5�RA�����z���s}%,ݭ���|��:R���/Z��E����K�#�"�� ���v���bx�S=�|Akc��)U,�p��
F���Vɷ��O�mf�h+noKik�c��9vln���"c
��)Σy�\	S++v���;r੮�V��U$�s��tz�ep�?Z�_GʺPl��	s��F`g�;�|A|�Y���j��e@�9+��р��>2��ZZ�8�'súc��4	/Fd�N��Z�}6�ڿM"�������8ӧ���Q{��*%Dl&�E1�0�a��l�L;f��Ž.m4��D��e��e�9�jR�{���s�WR�����/g�����ߦޓW�A�a�?���WFp`�U<�S���_p�C�o	� #^8E�%7^�>�Ĵ��?�H I�:�j`?e��d$�.�Ĭ[���dO��mm÷��wQ'V�'�'����]�\��js��/���!G\s���q ���R���ی����V>06��e��HR�_%�����F�p�_-���6���^o{V��V�������5��b�9���Z�u���˦���
 ��,xPtu<����SA������p	N�w��av@�bڼ4s���^K�j�<���5����;�0}���s�$����PK��T���1�7�<K��U����.�YmC���U1�]��v0�!���l�f}���AG��xE^�k�\���;��QI}�&a���L�������<�W�	�q�JV�vc��oA���}�!cM�x���GqF��d����U�Rb�
a�����zd����[�;��ո�g;$�LZ��o��ۜ��@��W����KΥ�< Wp4���uF��]6:�&��|g�m�����ʄ���U�AS%r\yC�o�8CI)n�h���(7�3�����)Cyg�È@��[�v�	�	a�ɔr[k@َKX^�jc�>�V�[Q��A�&K �:^hZn�d���=�C��2�2b$�r�&�\2p����V�o]X.��U�ϋ!Q�>^�8�Ձg�}H�g�B�w�o�ɯY��~!65^DSI:��o�d�̏�W�v�	� H/�ޘ#J��َٽ4�����u�B��	��ȭ��Go,`[b�O��$n*�� [�*�uf���tI
�<Ֆl��S񹵅���䧼ʹ���ܼ^�C}0�W�k�@&P�!F`AfPFX�XL�a�q넩��~P%�x����j�tҾ	�9W
Y���0j�N��J���k�_O;Nta8TWd��j��wY�.��Μ�v7�~����齇,��{��g� 4�w��� ��us���o����p���IsWĞ#�\�a��=�Ƶ�p�%/�Y���"�,�I)�GPyz5EƏg���i
)�3��B���&ڹ����E�9��f'�ظ	bx�p��*F��s��=��^ �������p�h�=!��ZO�M���F��f�A>�b�]4)��� �|�'m?��x��Α^a��s�CV�U��7�S�q�t��1��p��UL||�\���N��W�p%Z�{m�
���=�G�[&�p��Ԧ��8ܓ�C��K_���-�	ntg��#�YF(�2��+y�qKI}0�g�l�>u� 6�,����q�[��bi66v<�D�� r�3�(���T���̙V�I:�����|>���O;9��?����/���� �
jƒ\3w ����p�Q���|���.�!�LXr�<-;7X)��7\�N�A[	��W��d����{�1��O������5�ٛ=܀:y�Hq�rs��W�历�F{Ќ�./�\G\���2����D'�c+�"� V�@�dg�n]�E�c8|��<{�(J�S�}�ّJ4?#5.�����z���RE҄�W�.�S�rI=�/���>��K�6E�M�58G����>䀪l_�%1�1g��>1�m�;e�0��h��"�&��R ��nET�9��+�2�_LVyr���v���M}���f�}$<���-����#�E���,���[ֹ=��6��g����[�,�Du��0M&MMgk��k�pIr��j'��\���S)�X�[��ɭ�؜i�Sg)��a��C�$G��M����x���]�t{�VXC�@�X�'��X�=�����-,���qis6 xT�ʾ9{hsk�R-ؘ�E�)�3lnG�ȗ�ϻ���*�;�M��kj	�h�Q�]�ė�	���&Ђ�$�����[)>�y&@.�14�F��ڈ��I߅�����e��y� ��"7��sg��F��������1PK���d��(�j�(M6h��e���d�`a���h�ێs�&���ğ��z��{������IW*��8�CD�e�q�*��sn���&��/��ŪgX��*�^9�}��`	@��1F�D��̒E�'H����8D�)�U�{i~NdI��=��VF��2�J��z�As���'�kL�K�#I��q�����<�)�YU���1<Ć�(j�~�?�cGfm���_��]2���,�J�:�F[�~<�Fw�u�m�GW�i7P�K��,��剠*�a�Xo�Q���t?�]�&K�x���p�Lu0Q�q��[�iů�� ����@���������2��u�Y����H��55;b6���y�G�f�FN�/Pj�4���|��I.�p�#�@�`��A�n�ó&��՟�����p�pk�jЧ���}=Wo	�=��/�Y�u-s������?kf�MQH��nj蝽c�i��`����=!�
��m��"@��g)���B͉l����
4g%Mw�}�O�6{�����A�BRfe`�̛T���a�9'Ur�����*D`���=B3jE������1=�R�B@��8i����V�nc.�Sb|G��
h`������
���X�Vz'�����x�sC|[�Z���qrYQ��5�#���Y�W'����Iy��JQ�$	L�����|��$�1�^F�v�Y��U.���+���H�}9W3IBb��G�>A�M�	�@�ފ7�<b��2b����l|�H�
���&�V��*�>��b�'������r�.>��@���fY>�qq�Q�{�}־��D�W3Q�N�7��֐;q �\-v<?��ޠk&Z!w��j�ɫ4�����K<��mG]7,�]K�����%x���q����N�Z[�Ɉ���C^v�(��9�����!��n�4p=;~O?͂�e�����[4�k�������i]w �#k\�(��l	Bl��C�3���>�=����n?-�����D8iFc%�{ޘ�,�~[��쓩�?m��MCr�g>u�A.��,F�\�a�VF�E�{#��u�t�;�EP!y�)!t�ڴ�e74���m(�'�bh�4-���C'��Q�Gp�#�Q�>!�di�b{�3چ.�YI�L���]M�y���3]t�g���yc:3��s�/��7�$� 쵺PD�֤�����ûL��VM��xy���ɂk�cc�K���R=HXuw���)��z�7I�(�JUk�f���`(��@��	C!?2�ptmY4sV�32�l��Y��Է!��G;љ0�)�H��2�
;Є׵��^��K�!�n�Ɔ�~Te/�kL����t��6?ʓ2~��W	�]dE�^�ǌ�����q���"���>��k�ov����h�N�g��n�S�l�9\�|���O�L�iҬ�Ǯ����3h9_��������`��~�WB����;L �#( u.� m3�^/�#_��|}|LL��W�I�����"�ܬȔy�lS����!/	Z���N�y�K	C�10F-�u��U�	�'�����詡�ۓ��]�aD��* ��e���_7�F���>s���;��$zm�E�]眿bFz�M[���5���w�_{+M*�L�&�6u�c{��
�z��{e*S�)��3��a�N��3�.֠�1�B�����XCO�{�)�5t7����lU�&
�k����M�+���:Y��h���*C)-�7�h-�T���Nz�S��c��6! ��}��D�N��0���j�ftuR/GsfM,s{�Oujqn��[%��/W���R8��+�ڢ�W�4RE-7<����P)z]��J��K�N�G93P�m�3�A!��r�}��o���X��Z�X�%��|�v�S��>Ec�+T4�<{�!��vf��}�)D��8U��{'� ej*�������Ge��G����l���Xwnt� !���J+@�A�CR�r��-ט`m�7���Z�7,4�b��6�b��̉m�2e��S�>vQd�#�1�`��@Չ�-wM����H �7/��`��CG�V��5�1q/'Q���p;%�{����IC����PI6%�ڑ�Q�5Kق�@�,!��GnD,̾o�6[wB��z�(�-*��W>mKa����������M1%qT?Ѯ���� uC �i�w�i}j%�ȖX͓�	���=-`(\���Ջa�H~ǗS�t%l���ű՝K:��l9���Ys����7��/�
��b�X����'�x,Ë~�Byx4PV���,z�`��&[O�τ/��a%#%�[� z_UY�#S�a�噛Z[�Ն�t2]����K��(0��ݓ�6y᭧�5a�ugv	;c�_"p׼���w�Y��@���@�e���8�3o�Mh	4�u�"�[�JfP��oFH��[�����y.�N8Z���,[�t[���k��̗z$` ����~X���𽑆諪�p#!���y�Pe}��[P��6,������D��S, ��8�D'�؟q���?q�a4��.$H_��7�=p�����4���E��_y"ξs��3�ak<��5ޫ��6<���/�'C��P������~D��G�|B�z0��L,D�4t��U$�ʹ)'F��Չņ
m�a{X
�/���r�V�������hfaԱlIpjÑ�l�.�Yf�?������QZ�m�Q�l@�6�5�M
H�~ �M��{��̐�R��%ߎ��K�T�	]���S蒻�|��c�b���ݲ�,�'v���bG�㌊�ca����B�,��"�F��[��WQ���C�s,2V
��<	>̅���:@i�&�xn��V��}m���v�a[���uc�Ϥ=P(��N�Or���~*/s��>q+��?�J���,�9灋��T�yVIy�)�Τ|X
	��r꘥9�\���+��O |ƥI�tz8�6$,��_W��լ�Z�t��6���,P2�p��Z��%ڶ��UJM� �.y���>
e�(�
�R�#�V��I��S�Ѣ�� L���S�A�s�k���Bf������'!v���Q0�A�*�J�5*c���Iu/ �X9��v�������X���J
�r��v� C.>�������N��^�j�X��B��(�((��^��y�az�4���~�0..�Ћ�k!�)K忢�#�K�,�R�j�qcD+r��(P�A�vfl�U���z<+{@n�������
f^���a(���]�2�ˈ��w���|�պ\�u�H���Ұ��1u��@w���tQ���!�ǍTi�y�z��!d��� ���v����k���=Q�V\�Pj�>����KQ�"���F�
�3Y���$�B֬p�,r�R��Z�xE�V�z�,������7?|�9����;sf���9���1mQ'a���+��,�y�k�kG��9b\�8h0�|��;U��Hz�5}��Z��a�\>Ԩ>[ڲ	.\+ٓ1-�o3\wl="�F��$f��k5dz��5����҂�Lu٥�[L!w���J�_O+:�EM��N�2����e����ӕs��=�w����DD��}=�l�i����i��E�3�/z�9tN�aSK[��O�b���r^�͋]�Z�6r����N=A���s�YE�3�'�5�pۭ����}ׇ���g]�ۜ�{T�q�I7�7]w��9��.�7'�3{�;h����L���Ċkn�ot^r#�1�v���'�Z>9�P�s��^!����'���y3#<��Gl��8�~JB��nY^4����'ۇ�%l�7~�"���(j3��{㐡�o������Y�[G�,W�1��_�~�Z7iF��s�&Z�߾�9��͎������COo(h{�l�8ѣDpoɀ4^����~4��E���/��S���;ZR������ū�w"��[W�\'ۚn~�#{J�o���ЃKg�����X6Y�\r�:u,�bw�v�	�5'���p'��
�ٯ?.�Vp�
�>|N�dp��c�8��~`hh5�g��?oߩ�/�/�	�v.L�V���)+��|t��ܽw��40�N������WCb�_���;/<�����U/lۼR����˒�d�/K�J���9��	��̛��m~;蠲���B�&M�-����ݑ-a�F���x5:n�����|ʨ�.Ě[c�n�{��p��m�z���5���;M�Zvm}���+�BDЗ�E%�o��:x���u�ۍ�	�~�v��l�<G��%��GΩ�,��:7�Y�	Y�y�Z�<��Bj��ǛgI*��g��iߑ:�~�6��~p��E���p�~7�͋��g�[��1.j���h��Q+��v��4��~����#��ݩ���|��c�ݤ�|��{`I�z��gc�������t�I����:Z��+n�vT������l�ڊZߡ!�o��6�Jj~�x�wE�n���j:�$߇��Gݿ}7dL�fp\i�Stl -#��2��4Rc����jz��1���Ϯ��Q������rlu�S(z�*|����B���5�\I���n[91h˕vU�b���{L�G�
�	X��Q�3w�KRu{�yv9����b9�;%*�ýs��5���}�pWՍ��X�B[�ӌ	?�y�mhjm};�%>F~�k�'�oD�NMl��t��pvE�M	k�Ј�����m��m���G�94��k��߃ثMK���O!����y~���=��Z>����9��+���1:����]���)�e�?�ﲬ<⿪{��6o�����5.��̍�t���P~������hX���OCj���H��<OؑX2���]�1�k�$ֲ�����k��-p�'����q�<6_U�e�����'����(]O�_{�P�xa~묰����/�7WD�yģ�^9�d����v6�>�a�D��z$&j�q�����|"���V�Jެ}F��?�{��3���,H�߯v���R'Ϳ��oZݵp��;��4E����
}_ ��P�3�/���������+�)C�?�֠���~�q���8����%�쌬\�4/W�%����̃��$���9��L�$0K����/q`0[�P�#������|�8�g̓�K%"k��8�AW���|d\$�ϗ������|�X����t=8��<ė��%�H�3�s��~��E����\4���g��g��z�}�}H�|�>l~�u��G�E Z
ׅ�;=ox^NP,A�Pf�L�\/i.z�l}����ϓ�tF�٬���Z,u�&��JK]L�����Ơ�6�k`���`���--�3`}Jԧ��jP�-��ߖ�l0�>0����z�
�	������O[�j�����|ʟ�D$��TR�A,�Hb0)D�b�$R�t���D���4��d��`��d1��Df21�Y(�>��0�5�ń��8,�c1�,@�f��D"�B����8<Oa��&!�Ȥ�),8F�a}����c1!k<�/���������ta��K�R(��\AjjZ
/9�'HI�g3��t!J�0=#fFFd�+H�|)��g��x�t����\������|�'��3233fd�e&�ǮZ��c~,^��A�X�d�qy<>�1y|��@,�/ ����&�Vr2L_ ����y&�Fx!����z��~$f�<L̓�l��f��8G HI�sX���LA�@����F�����G8.\��ID����Dc0 ��Y	A0�}�F!&&��&�AL:��H�&�Ȅ20�棢>,�ޏ� �(T
�I$2&��_!��db�L�lt��GB�������}�?8a�
��h_�x�GL��it�*&�G�Y�`�>�/cyQ��C�����}��98dߧQ���7\q2n����	���>�z���o�_\�-�ZSV^Y�Uk��zCu��Lg�� SSQ]U]Y�єUV�,�F��Ҡף4��a����������a_���S��VVUVW����0�q}EY��h2�4�����V~�
�ɇ��)�U%�R�ZSZ\�P[���,*)--U)
* 2�U�yL��X�Q��Z�?�������"�r�
��b��W*@~�V]�P~��"�
�Wl���U��7��~�q9�d�E�bE�� �'j��Q����}�R�d��D�$����^�RE�´�[}J̇�����x�����D	?wr�/Uɱ��c�<"}��w�zj�/��Ǩ��M5f����:��oľ/���F�{�7b����ؽ�c>K����w<�G��+ly �����&��7��_�Xg6���E$��z��;/���c|B��ǅ��Ȍ.\�B5�p�rX�?�[��˱e���[��˱e���[��.����TREE  ����������������                              '�                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OCHK    ?     `      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �             m�            �$�OCHK            �4       _Netcdf4Dimid                          ��  �p��OCHK            �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   �XOHDR/         �       �           ?      @ 4 4�      ��     �                   �8             shuffle            deflate            ��     �           ������������������������P        _FillValue       ?      @ 4 4�                               �h5�J            m�             ηf�OCHK            �4       _Netcdf4Dimid                          �ΣOCHK            �4       _Netcdf4Dimid                          ���OCHK            �4       _Netcdf4Dimid                          -SPOCHK            �4       _Netcdf4Dimid                          9��UOCHK            �4       _Netcdf4Dimid                          9��UOHDRI                                                 ��     �             ��     �      �F                ������������������������0        CLASS                DIMENSION_SCALE )       NAME       	          carriers 4       _Netcdf4Dimid                            SVOHDRC(                                         ?      @ 4 4�      �
     (                  �8             shuffle            deflate            ��
                   ������������������������P        _Netcdf4Coordinates                                            8���                                          x^m�mLSg��am�#�He��Y�3�%I����n��JHh �"Z��.&�:��#��mFL\N2IA>�%:ޗ�P� �T�0*P?�R7��as۟y����r�I>v*w��o@f�<J;���r2��
�7)O>0�jq���`7Z��RУ��o�,��0�йkvX�|��y��줟ө($Z��*-V�1���=Ӫ��%��_��b�����VR�T�1m3��!���f} 5-�[z#���{���1���q���SEl�AR蓼ϔD6����y�$�:��.W���ɤ��
bNՖ26Z�	�KT�7j�L)g��-{K�ڪ�*FN+���~v_���h�RV��)K{�`�h�	��R����ʎ��� �aM��~Zk����-,��,��{a�Ө��V}(0ŗ�rK(���E;��nXä�����4#��7�y\���{_����!·��2�&��Q�&�}��K����*��OS�Qu�ݮ�'vտ�pVx�]uKf�Z*�L�C�d�I��6��B�7�!\G��`��_��0��p�vո�� ����Ĥ�j��m%��'��������q&�7k��C���iGo�V���Z��Li�6����c)27�;4h���P?,7����n�GC��@�E�vlӾפip�h:��(psב`h|b��2��X�.EL�_L�l�f��fĉ�!Q�O�������%]dF!�1�:�H}���!	�[]�E߭��׋^��\?�F���:z'Е>�]^ �a��L�?7�ƿ�/󏝯��	:ڈTREE  �����������������M                                      |�                    f       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^�w<W���_-)����w;��H�B�P!#��R���C�T4hH��mM-���\�Y�9�������?\��ù��z]�:'���5^�Rvq�����ՊC���40獖#��|����+�t%�i��B�[�Y�@����W�W�$}�PT�C�r~~nWm�W�SZ_��d�d�Ɏ�Jxw樼S3����ܬ�[I�m��֡$��7-�\�ŝL��!���I*S����0�Õ��d'���L�Q�0�h��2:�h ОT1�����WI�/��L�� ��~Y�6GŰl^�@�l�@*��-F�gOI���u�A���*���!:�<N�*W����Ӯ ɷ������w\`�>�b�Tx1�1���!��	��e�f+��2���Z�d�|�ˎi�P��Rjz��e��KO��<'̯":ϐ+?j�6��8Z䴙�)�B��9
'��uDVF�y��k���e�� ����e��i��wM�*$��J|!�7��:�0�?즨���2�xt���~��k�+Tw�Pa��N#?���knV0����2��g�U�P���
���u�~���
�~�9L����^�R����Z�_~Էf�-���ڟ�]�6W(�%>��鷳��Ƴ"�0g���~��d�,~�%c͗c�s��R����X��ձB��*�O�7�P(���z0��D椆\à�`nG��`���݄�vB��� ����;h&�f_�f�=�KT�|+�<�I��! .h�0��j|	��[HH'*������v�zJ�ž�l��4-�W�l�r��m��
���䡷���ݚ!튬�a6O��?`��5W����&���G&Vۻb���{��
DUJ������oL����]y���V
;�bB�W��R(��9�	Z�#�~�\ϐ��1�q幕�sP�_|�!�R�D�;sp��ϣ�B��5) =��֩'�^UC �D2���ZzqkD F�B�~�{�x�O���~���,jEj�-Vw�q���̐�M�Snm�ֶC�=rj�ٸ!6�z�2S��,��>rsӼ�>`��LS7��K�N`�6��b]�� yo��Ru��˿ض�M
s����fH�'�ס⠣k#n9����ّ�����w���E�r����m��X햵�c�j��� �Mk`��p��䙣cJ�V�/F��ܧD���>�8�����?�ڃ�բ���}��/w�}�ٲ���\�w�+C,����S�.!��ٽ8�u��d��3������P�F�����K򙄶uz�k4t7�4}ИWg�&}"�Aw����y�mB�c�x�'Du�텳�J��Wr	O��OF~��|Tu��V�.��\]�%����^ly(JD���@t�1 \�
&$��}xp�������ſ>}2T�VI�{n�c����[�O�TPmB=���u����.������pjX����`Q��I�1cS�-03���7n���꫷y�b�E��I��ԝp��
'�V]���)�qpt:�a�UW�8�~��n��^���$�U�r
�B)UTYm�����������8�}����'�[%i�._�v,�Y�a��nM�熔����yj}���y�[o�a�7=�>!x�ǗoT5���rmb���X�e�MkkR��$�Rd������Β���򱱃b������or	�ݙ��uR���Y�� _7q�m��z���B�ٽ^Z��@���s��#�G�� ���w8��k�'�2`ƻLT��Nk�s����L�1MVŕ�>�@v�gD]�Y�O��̼ʆ;݀㤿w����/B�0�{��x���ϟ�>s(���?���W�r��r�ʓ�?O؞[�
�B�P(��>���l��ȼ��ʥK[Z�a�>�!3���r~a��ǒK:�<Vֳ2��T$�k |sc���s(3%A��U ��z�0	qb�'~� �fg�<D{�TZ�!2�B�PJ'Ҍ����9X1����^�󣈕����G<�J٦�ȏ�"��µ���O�	�f��FFn��oCۗ�4p81>2Һw��=�X����|I�N����&@\J��M�V\T�}������M��sKp��PJ1�������$�����_}�Ok��D���-�3��[��h1d�Y�Sp�L(/���<�]	��.!��ޖ�2r����U���5.+�7 ��Oi8�C��BLH1�Rf�_�Ƕm�wR��#�����+����P�t%rI�Ѥ\�$�Ǣ�II�K���F�k�����t�C�<�������4`+{j$ݖ�R���ɣ6�{��]�a���|pS�øh�VޔB�P(��Y�鍊���y�u��_/G�Ǥ�.A�O=q9E&�g6��i�!0� �5y1��@Z}���XTn �B�Ȭה�?�@4`�B����<٘�Q?h
�B�P(
������e����g����N.�C����tл[ڄ�Kz[v=���7�gW>έ	�3��(�*�!7����1mO�o����V�\K)CnT����!k�����v�I��"�w�&�c���ϝ� ���އ6V�y_�Q�G�e�޻U����a@4w#�d�Q]����]��p9�5J����ϧ�������]�+�F���Uހ���:�,�~P��8|�C�� �|[m��Ʋ��;�H Nޙj��1�r�B)��#Z�Vr�K	�g�RuFxq��G ������� ��1�m|L�Ō�q����+���8���g������TtTB8�/aO�p(
�R*xceT>M�Lvؕ��.���9�l����Z��� ԇn��F�ʵ8�轲p�%�bˆC�p2;\��[{�5<�ZL���v�o�K��V����>����G�-Z <����Ķ��#{0S����p�`ŀ� ;��E��T�|�a���ܻ~��B�@L�:�g
�f#��=C���Ǣ�J+�&�ceb�=3�3*�ٕ����(e�����t�`��
�h�`R9���+�c�N��������hp(�M3^�����q��5��7�Z�>W��b��Cn/MvqE�[�useg�^ �N�g�����Ie�Y���¦輻\i�tJIFP�Ҷ��ǔ��!��l�\O���;�p�TJT,{�3�cο�O#�y,�0]�UH��z�;j�A���]�T�2�8�C2�uKլ���j�wð�{�W���h͎~�nv;�bnxn����޽{����m$��r��1{.�,�>�Z��SWao«|��fW/fuN������9W����'�t��{�]���8q�q��z��Yf�����	��)|�/&�ݜ�#?5��v�qj9k���{3$��(J)f�8�����ܰ��6r!��u�`��9$�\g����I��+�2��'	]��_3@�q^)w ���tL�͖RT}����O�әΖx#�R�YPg�"��/���g�d7G�����:5y��p���]�61f���L�C�_���������_SW�\���.1q�E����DU�ee\����>��+uW-�M��4Kh�����!�R�Y��Ք��L��{*LL���9"��$�����o���5Z�1T��J?��L��8%��F��Ȭ��o�s`)����)��"\	>��vx��g�Ud��m����O�;o�4������"F����!�+x����oP2���1?7�gWa��W��HOw7�]b�x�h�S�Kx�,���{8E7�m�ji���C ��	ycp��㻕�޺��-��w�w�@�¾�)㯭o�Z���������d� �'��FW���[��?��۬F��lX��{�c�<�yk;�G1v�~ކ�u˝�nӥ�)|��߱�}w\*:|���KV�8�W�ԝ�.!ߵ�%����g����t�8�4Z��6��N�=:�
q}7���Z�N��_@yhyT2[X�~����0��s�ňZsg�.q��o��nq��G��ΚNy���^Ns�ꗍ���^��f���/E��{��a��#�6^e��y���kέ}0�m�X�HѾbm��he�U�5��zԠ���T��ڎ� ��[h��u�MA��k��ֱJ/ą��M�f��}�Cv+�tg$)�ݫW@X/����:�~�zp��ǽ��.1;_v�-V%L~R�߷����F�ƅ��p�}t�h/�潶*�b�P(�҃��jz�j�ի�s�y�)Yy}��^�V�:�����իS��n�����|� e=��+��U�e�,o��6(�Ȅ��p���e��{�q�G��nޕ�V�2̙�Y�Ś��y�@E����s��&l�[���oT����!�9�&���/� �KF��|\aצF��p��V���#C��= �Ǎe�(����*��-?�I�vM<[��K�E$w'�ĩ8N����=�[nu�!����Y��[T^�)Y*F�<dc�1���p��� h��s�"�e{q���mW�`�2e�Z����x~�Қ��%ϝp��TZx^�Y[�LE�P(
�B�C,����*܎4�+�r�˯�ƏT�O�r��h�9�e�x��/@u��@j�
�o[�Y�g衅l�q(�/�fFp^TĬ�h%�����}o�P(
��1�zk����YaW�7T�{��Dm���m�ٯc|�K�l��@\��0���s�is�Òe�""2�DD�N}Q�2齏KjDD�s�%&�����bUF�j��/G�@�a c�L���r��٦�"t#"�	YM)�2�)�M���3�9�:i�4�}W���ڀ��x �|��E�7��j@�k����5x�p��3`�`��ou<!d�n�0_;����.�س�YC�����{��fϨ��L�����BoU���޵���AB���g444�]Q��%��������
�9��`\ȷ�@c}�C�<Pŗk�5�R۬��V��!�	�.,�]d��Y��X�C��qӸ�����<�)
�B�O�+)��9�+~��S�rM���2)�ʬy�4��� l �\<I(h
P��X7Ȃ(�v,pv;����έ�3~z;��)��b�C�=��3(
�B�P(���Q{��нK�z���>,��!��{3����f�z/��W��iѧ`m�nu�T�E"J�Y(l^.��eڎj�>�G��%d��	�^�v�Xw��{�G�oR��#�	v��Ӈ��� �z��8&&О��8�!ԓ�.ù;��ph����)Ѿ��[�G����Y�����p{����Q�2�{��3�7G�	��uZ��Bi����Ye[�V���\��J_�ϻ��- _{�L���R4`�T$����;:�S���70�Rf0�G�ތ)e�-!F�J0��l1�?L��ϝl�l]���̣ʟܔ�}a�|��ֽdi�g`����ü�G��w1v�i�tT\�O�,Ϟ���)
�RZ8xâ��͗l������}-{ұhN�͑!���?�n9RZ��d��:�h��Úr��q�v��j�bc3���͎��!�6���(h��aw�%K�q�DUB�gy�,#��Zi�̍�]K�ɶ=h3�)R�M�$�����[(����<Y�� �vq��g��D	'�	CC������ޓ*���w��o����'i�]4VJh��@�G][�'�A�Ǒ��nCqHC)C<:��=�%�C����l��b���Ϙ:�+��e2�Sd���X*���0��>��Ë9:R�����̑��	��ȡ�q�M�(����٭����R��(�Q�.���sŹ�8
F)�b��G�2��=Q����\���ԖuD�F�|�k�Jr�1��w��K�u��;�Ӭc>҃�'H��
�̝�Hc�� ��Ɣ2CO o����X�,��JS�I�׵Y��{Q@�����t��{csz����>�n9��S�M5�G�����Y(���ۯ�3���}������R�1p�����.5�*��6��r��߬3�bԔ7I��B�")�&\�T�ɝ�>�A�<Q˟�LM��4��$�/�B)��1[�f�<�¦��V��`H~��u%8/
�`��'�< ���~M�Q�af��-Q1TнO$o�y ۝���zN���)��C��BL�ی���)e�#u�17�I�\����ܓL��f(8g�B����=�A��a6���˵��ϟO~�����6��bo+��s�e[֭�DU�^�WP))�M!��s�6���ś�;�Υ+���sG�D:u�JYj��W��:!���4�F�%<�c�:�1t�����$�˯ �
|��̅�	G�w ��zj��k0��@2c����rX3T�<�N��s�_��ñ��W���\����$B�g����G��+������[rs]Ϊ����3<7���%f�̬~Vb]Bh7yo�χ��;�`��u��Y�����\��>A���|<�4����>jQӑ�ON�N��!�i3혘�=cb��t���I�c��1����TxV���!����>{��~����3�P(8s�$]f���+�<&���LW��!�G���Y��ëoЬ�Y�K��V����+��1���V��;���b
%�b�P[��������>�̉����ދ�6%=�ﶂ�~ѿ��Z"֍?��Tͮ^�׸�[Ƕ���'��k�De�ӸK̙������(i1"�6��5 � \m��l�z��+�o�U��{����w�w�ʓh_v�pb�"�.[��l�Lc���+|�͍�9'�輌\q{��q��:�����ݕ�$wSX��X����S�)����\~�8M�%&���ɱbUB��bP?)�m�iG�9W�W�r�m��:��*	�@�P(�����Gh����թ�f���{���[Ӈ�\9���~Z\'��z)�3����醔����z��A���6�3EM� g�"pC������s��0�������8�7�0��Ś�Ge-,��C��ުl�1�:u����+���_-�_LR��� �9�B`I�+��r%+���m�x)�ǂ�H�DI�t{ 8ȷuC�v�;�ef,iG��=�	e6�����l�D� �+m��ǔ-�m�{�6!�La����^�?���g�"������{K�Րi?��()�`D���4ܒ�@H�����n��c
�B�P(��&Dr����9I�u��\�ѹ��
�=\�	���|��
��*m�Q4D��8�8C6id�U���L���6����	f��7��g����� �N�P(�R�(ˑ�s�կ�9�5��M���;�Z����&κ�u����%w��1b��)�@	FS{�Um����f��n�:�HּT������%f��6�Ī���\�L�w��Q i9����D��6�cml\s��=�R��Z�v.�P�O�f ب��q�����t�>2c���� �a*(nm3N��ݘ�	�)�WG�g:���F.��;��C�,��eT�%�VkQ��kW�)������:ĳ��_�<�1�R&�ݢ����u���qO*��'� �����z��%KB�Ѣ��l�͝6���T4r�����B�����l�s�>�K��Vz\�N�����A�����yn�'�upQ[��B�P(��a��G�S2�_7E�X���L�q�k�-b�ؙ9����' �w�f��-!?W��������6�-��W�̇W�Z�a�	�l4j,�B�P(
����Z���r�?�&;;�ң����?��\�gϣ��ҌZ���#;;���tت��\ � {Q$�H�b�\�;�i{�j�Ʀ�[Y�\J��l�l��'���n4L���7��7�>؇�(n�vؓ�	�Ť��?���+��KH!b?vm{�!К�@���P+�a�Oq��Hd�s:��}SI�JJ��Cǎ�:r,�����y�@�T'V9b�E>�@���~_�/�|�d��������	@M*�w��ø�t�E�U
�l��M� �Rs0�8��"|Hm�D#q��Wv�#Y���~� ���q�?��d��"��d�;٘�yw�GA!���Fq�v�����ϡؾp��\�c3vp.�
�B)u�M����m����V���!�@����ZO������]�x�bܫ&?��+���C��͂�W	V��5�7�M���h�"�K�s��~"��t��K�5a�z��{��EضYЂ)fʛ�I�p�Y�T-�ǋ��������%]���L��N� �t��ㅇ����.���Q��{��-G�D������"������p�-3^ G7�{r\*��-����K�/(� }�3J������moKR�`>��� ����bID>��j�l �����`z�7>������,�9���w2nw�[_^��] ��بKg+!,�,b�u"�k����)*�QB��� ��K��Ӕ'.��ޚ����>�8s�k�}��16�B']f�0�9W!�?0_��.�:�2Q1L*����G3+�ܴ�_�G�M������<W���Z?[̥Kً�:�͟:...N�e\jX���ϑS/%�!�U�S3i0�	�/���i�������X��\�yh��֦�i����s^�Xv)<w��V/V���������d@W4�RB��o3?W~�����LԲ�qq:qw��M)J)Ÿ�>���MXv�\�V.$�I��'T��3c��i� ��@CvayjM���
`VTzHʘ�;��J�2�M*+�<s5�O&f˹�BnJ�f�@��5w�h��DӬ���Rr_Õ�(�4c!�.H���R�/���?`��������]�v~��k��k�e�GՁDUJ�$¾o���c���j+�4��PW�I�u�i�B)}��ĝ�|p$W,��S���O$�/�t�d<wm�O(!��Mk��~��Y��2O�\� �MqZeM ��
�����3~\��EǍ\������2A��-��15��w�|
6,-��Bnߔ�ԕ�W!{_"#*9Ņk�����:��/�_q����vAAh'�] )7��#.�?���ԭ�b]��Y��d��>o,۶טM�
f\���C�<��:T����Qk]�6�����Y�}��q=#��ٕ�KW�_=+]]>}2&`��U���QEuuS�����r���gax2j�̾��+�m@b�r�XvSwė�֚�߶�a�����Y�pp��Uĺ�mOUe��(���a��N8�t�j��f���ԅ��ϽW�6����j��{�߄�T��8���^�%R�?{�~D������J��%,L�ݿ�ї��S*T)�������*v�q���U����tX���D��7􆎇s�N������W�O�m5�m<�m����*�?SI3
�Է��s�������/I�=���`���.Uo�%.��6m�"ew˼Z9����[`��A�y���,t�r����[u��{������3���I�X������Td�ȅǯt}�1#�僧�m/��.��.�d�)
�t����{�=�ݪ�9.56F��P'�9�~����j���k2,՜��e��K��ZH��h�gҺ�j��+����Uv���PN.�_^�Ѹ<3F�Q5��xy�wX�*'$�5)����22�nX����������j�R2�fx�,�8�q��\��Z������2N5;6r B��L�c= P�
){�䉷�ƍ� &#�� ��^Ƈ�t	��Q��Z��P9�f�"`��+&�ً"sdu�l�G��a(��BcM�`!!���ĝb�g´�a3�4��t8����#��fR�fԎ����ٌ����c�;fV���\�bl�rU�uI�P(
�B�oC����s$E�/���Jx:��(��L?�t$@(��]3損���B(+��{*Rn��WO���谛��`�H�(l���ـ-���S7d��a};���/�Q(
��!ӾE��X��eӢ����C��D�����GU?l:�X�F���m��{v�Y_��[6yc���d���66�_��̩f[`cSxw�	��+Ī��:H�L,P�px�*���eĽ�l��l�ll^�A8)���~CUb� ��v
��
�qj��|w�]ǐç�]V lNS���׾����7��h"���q#[�=����< j�b���s�#h �ǥ�{=8�mb�\r [j:q����B)+|\��;A���5����l�f�h�ѵk�~-�ZML�x�64��hV\�ڵ��鳌#h2*3۰����(Ɨ_[���%����
��k��2�ŉ%�5���MG�d�rZ��	n:�.���Z�P(
���j8���+Nkդ�/"�z *!�l��-.Kv�(	<yF�[F�t���E�����9�̒`d:��y?�A8kAzl��G��P�ko���V�-(
�B�P(���GG)��{j?x�`�ʔgr��cD%����7�Pa��X��,p{ ���%�߻E��R�1�P��Ҙ���{���}�\K)C�;��f���Y��ݒp�F��`�ps���T�TPɨ#��V����CX�Ç$0���b�k�:��|�*
�=�o��۝�����>f��ܭ�R6�������������x
���%���h$׉8�K8rt���l�	����7�'& �R����3%\<��ơC�b\ 
�RZ銏��RY&�H!�>�腮w��}4�z��sC�0C���XtE��<P2t�4�f�{�v�� E�fƭ'� ��R�B� ^a�B^.��F {rB*�P(Ji������5����?$0`�n2X�g�+g��8-f_��a���ؠo-yῥ{����p����){��wkw35�_5#���w;q�,v=�dU	7GIs=	��fy������l�}��0ſ߬.��!d�TQr�.2��Ov��,\)��d>%���*�
p'羨� O���d4ʛ�y�q��6[�p2�	�%���I�ԑt1�,��A�����-熱��n�`�>��		NO�)p��\;������F�h&D�2�ŭ���s\}���/��b琙
3���YF<�E�8�mz��(�w�R�.>����MUQ߁��̗+��'H|N�h�|֓Br����u��#.jv�ɝި$�hXDKϪ8��=@���\�$����i�a�2����R6��{،>�;�:g^F�M~w^<���O�4���[�NΔ[u���fdd�}�q�D�G�o겿,2U-��{�P��.$�:c^�]�B��j�v�����R���o�/�x4t�����ƿ�����ax��&B@Rk���
���nΝ_���d�ԲpIBFF�O/�|2
�43�,��r���&Z!��Q[���>���|�yeK�5�;z�] '$Q��j|�:�{�e��. l2�3��7ͱ>�?����OM)Ӽ����8�I����<ؓ�]����k��YB�1��i[T:]퓟$���֢Ř-�]u�Lï��جE�a�G���V5!�R��{~����뤻��wj�kq�P�jx��B)S���4%b�?,vД8E�>��gb[�גd�;�as8^�oPbi�k�6nC��@y3�EJ�g|���2����`ʕ�y̅�m0-���>�O�u�+1j��U��y)ͳ'�߹�s���Z�,���U�X�}ϙ,ı��:rq��IYY+fe�ߙ���=+��#�%���V���u	U�u\PJ���-XͶ�^7������0_z؁+�V�2��_	=�$$\M`��ɠS���͒�f��W�v����Y%!AE���9�,�h��H���aBB�o�.M��V<�?h�w�jE�`U����<:u�̮�~	_�`�mf������:�����ڃ���u	�)[~`�:�9�~B��1Ç�G-�=FG��fR��T��?�\���6>y�On'2�گu3�w4�{#��D�nx��0M�8GIS��#Jl��_�-E��8���wF��v0K�]b����}�(��[����|?�٬�_>^�s[���nu{�q�O�����bT�ET�i�'�4��w������4��ɓum*=�H&.�ؽ��$����>&�Ւ��,(ie+��ɓ��N����>����ԗ���?�w���:q�X�Pwn	��M�F�̡��V9���pۨ]kO����ZޔB�PJF�s�o�\;o��y�ֻ�u�ߦ�E�s�R�b���ʜ�y�����C5AJ��l��漷�S\pa0��񔍛J���3����/��bf��x$�h��+�T��5)���m�����>�,��<���s�Q�����*�J��/J1����l��҈}\Y)�h����}�9�K���/��\�:0�Q{   ;y9ѯ�^��|ǆ������� �������]�z����a(���o�6��t���.�[����Ѽ@��0o��������ƅ7 ��އ�i�s�#=����3������#h8�;�_�+_ڤP(
�B���pr�\��h�������-z2N.��S	Lo�P����0��a�LXY��:�T�>�},f���g��}�A4Z�g�&s�3��ql1z*/`& �X;���}�<�]�(
�R��j��[�]��%�:q�F\���kDm솼�݇|F8�E�����'Q��c�R���%R7U��ٓ���?$'�o�4�y/�����Fw���ò�X��WN��4���Z��{ұ_��޿�+�M��$7e�>)iO����j�$b�5�9d._�#K�8/0�*D��ף\n���D͙ަ�[t�Nf��*�y�� ?�?�� ��� ��A!;��*�a���A�-�D���!*�<�y��w)�2Û��4�Y9�$�����M������.T��Z���󑵃����+eL��VFB�n7ܧ܅O���+Yݬ����,�	�<�`�Ͳ�މ��7f/걺n冋��'[3
�B�P������Y�7y�@[?o��\rc95�pY2�?�=�2?�)9"`�Q��(����I��l� (Bc�Xz�h��>��9�j�ф ��&�ˢP(
�B�P(�I�le�������c��C��)���2�y��wG��f���:������7<+�DĘ~���=�0mm����Jm�.�R���n�E,� ��TնӖkD��/��J<�1?r�	}�W���7?�'��� �������8����(H�'8pͧ��:\�vb�2&Y��iJ� (e��cǖ�>�cO1<%>�.xEoQ(������:�C������m*o0 8=�y3�n,�_�����?����f�C�]"
����D�Wr�
�hI\>��F�ҥ\i��V�h�Q; @�o�:U���2�[�n��!D�g�+��q� ,���p�/(���T1C�k\x�wR9�B�PJZ�ή^���˕��3N]Ji��9�`�ι�����ϯr����}� �2;��IF'�G�,i`�'�\y�w�J��M�M���W/���K�����U	��Q~E%Z��B���n�=ض>���1r��ʧڒ�:"�?%�z�m "�9re� a>=P�&I2��L�6n��U1\a�\$����6�:��0�3>�jc{A��� ��l������S\8y��g�PH�zWd
l�rm�dW���H[ ���D���/�8���8t��;����;x/��w��k�\�Ӣ_<S
�Z��b��_�y����Y4�E���l�l�BM+�&�۴���'TϜ��A��q {����X��$a�1��0%�h�eʘ�y���\4"�H��,ӱ�(��1/�V3v�����(e�	(�ί�֩�)�{��������o��3+0&냚�]���F3�o�oee��ʽ������=���묆W����αBr�}�D���B>��K�m,N�$T�p_ng����Q�]���֊aR����! Y��_z+���wV��<�x��M��Q�W���ζ��l oJ�PJ){�¼mh�IY����wץ{�RV�܎�s�(Ƭ�ь�j��ϣ~�fv�����wJ��>}��� �,��E�II,�m�����RM1�8�,���K�
��ng��4�3�|k�ϒ��}��k%���Ն �j�5�*	���TPU���}/�k��xf����{dYs�9Q�R��7�����W]z7�ZpFu�Q�����B)[����u]�S3� w��P�T�mt�}$AP����ۘCVk|��d�y��g��AJ�І2(w���:�����`'�Z�E+�0,��lbR;����k3�Q+bz�a������T>K6`A��Y|���1���Ԙ'f=-�Nr��\�f��Y[�}̣�?T13�o�����.�9�#�%<T�u��o��KT�fۆ��r�YυS7 \��+m3g�ڲ��23�2\Z`��F}��K~����73s��%��=G�G3�i�I����v�33U�O��#�/�C����Ӽ���Cb�C"otZ��Ƥg�<s�����VO���|�V��s�]�zb]¹�������e�Qw5*׾2j]}2?J-{��=u��j+�N���<>ĺ��V�Ԟ��Kue�C�~�}�e(��D4�%%���i�U��1��լ�Wbs���ӸKL���3�?�%"[�q��+8>�k@�y��v�ν��Z��T�a�򚽮�[!F�&|���'&��vx�4��Aυ���t����СΨ�ۇ*�m�c����iU3�����P���җߐ��ҲC���^���[�z�K�sM�;��]bj_8�X�p ��l$_ݟ��M:��ܥ�=����9���Ui�p
�B)-�T	�u����V��VݖWQ���#�~5Y�'m�M[3�5
ǣ���0)m_�tk?�y�w��m���2�ݝ����_^x��E �GZ}��o�e�.cNkRL��� �L�ܛ��I�Vl۷Wlm�o��@犜��On�\)s�gX���l�S�q7�D;�,8�[�tf��+B_����r5<�؂�0ã����;���.�4�H�0�c�w��q�C(,�"^�3cJ�_���E�g���/��~D�ߎg+�30W�O�:�JMU���v�0��x�Hf(�[�S�G�5 t:&�6p�N��8��ҵf�C�B�.�B�P(
��a'<s�xN�^� o(�Ps��4������0�[3�l�	��貲'��o���� #�<j���P����̣�i�����X<�0�('��{�yM+����B�P(���c�ַ���Zk�ʉ�7�6�>����M��4������<Es�T]B�-QfV2�͆6�ޱFE�;SE���Ŏ.kV��[E���3w�ݎ Ve�=�grt��L���6���.��
]W�6=�k**O��j�PJ5j���ׂ��lk��XX%�s�x>�Sn�p݈;�KHk�����\~L9E4[������8��0:��9!�0��6�pw�h'��չGuA�!\��
ؖEd���K�3��'�RV8v[mrڭ������1��	�A�6�aÆ�O���B��"oh�pv	���91����'7"���6��Y�y��5h�C.�v��O��6���`ޙ�z�p�E�q����H�P(��*]��vV�������m�`�c����6,E	fCt@9�j��� ��6oA�v�l�<�oh��ZА���]X�Sx�j���z7����P(
�B�P(�=�fJ��S""" �{��y�r����~~MLi7��݈�������0S�âH{,�є>���m`�P7b��K(e��uJp,�	b�%��}b?\N���)!ۋ�E�4e&ڿn�U���]`O����C&t�'�guF�i ����ح�����9��8hƋH��:5�A�)e��-[���̄I]E]oM^�Ի^����U�:���Vxn���ڤ���2���ڊ�_�T$ |������^tx&	H�PJ3�"3�ά0�c��i���v�k��� �;�rf�b୮ �/-D�B�Y&�o|���蒈�̼�Q|�9L���`,��0��q����>s_�Ӥr
�B��\':غ�xxyM�d�l��9ɜPN��9e���42��݄)^�	^��a��g� p��r���5�u��(�	N��^C�.Y��M#j2��a�9�vd���1�e�����˙)��M�$�y���� -����ة����D���\)�8�4A���d�� ����t���b�ֳ�w�|Z�$����l�5�rb�o���@�Y!ˉ�\���\�(e��͹����Z��	�>���T|TU//3���� �E����P�(VE��1�4P�8)���@�3����(���J�����uPyMh��riR�ulŊK�P�"Ѱ/������s���8)�q��~�.���,w�$-<�!W��p:�o��*KY8�Gw3#�sx4��� <�#�]�!�ɜ�ƿ$�f�.g��,�r���F�&�G���DGDEGG��G��D�̛���<�7p��������訨訐ɲN����F�EG�Ϟ"�#���E�x9Z��S�>��<�F�ˣ#�C#}���	�
aE�� �����dN`	�T߀Y��Ꭾ�aAA�ᣣ����D���7�P(��]�p�Z�U���>�w�W���^W��r�2����Nq��)6pan.����1�݅�y3�=�P�,x��W'H%"�Q9P����|xG)��|?�c��7k�������	�g �?��=�	��D� �.���]�?�' `v@����T��Cǌsf�s�=�x{zM%�R&����3�q*<��9i�|S�L��M�)e��|t_���q�����GJ"�����EE�P�#Y8�t�;)DAjU�$I۞
z��ȫH1^@�{5��  Og�T��iż�)wuQ�"�}u�?�R�"*���&V#�t�S��0�����r��P>r���8坙���}��?���j����0CCf�����ʪ�p1��� �b]B_�����0��g�vĀ���L�Y1��_bC=���<}�{��L��j7�hh�Qc-F�{xz��:/I>�ñ�^�o���a��h�A#�F�G�:���c(4CyR��q:K���,�'�x����5m���״i�̍�r���>f����X���,�� �4����g��紉SPKo|��>�}�~��ǌs�8��f�����pB9x��	�GY�W�a2Y118���T�-2��7��ĉ�&N������7������w�k2��`�#{N���/�c�^0�{6݇�c�?���j�h��Gg{�1��e��Q����.X�$6*$d���y��/zQ\<R,
Qf�#(6;�'t��D�	�]͘/�ph/b�^�8<h^��WXHx��KL�D�8�*a��y��8����ExΈ	
[��.^�(>va|���)�B��&,�
[�p��E"#�G���n�^�QQ��-��_��:#hR.^:3f�W��L�_��c�'+$=�9=G��/b�ϙzXhxxD�-�K��L�&%�c��K�����������E��O_<_��C��J��Ox}G�ns�ï_
��~����Z�\	߫(\yﾫ�����
ј[@]����%��{��|��I(_�� �  �R� \��VI/S�ryx����i�� ��ȭv��+!��3���"�OΒ���qB�p�ƍ�O��I�0�by���?2�?�������oA��`���)�zy�Y��P(
�B��w!��+s��rjT�,KJM�>�ȅ*��eE��(F0�Ed�'��=��v��r~��P����-
f~��5�ku������5k�ʜ2�Ư�x\�<�<��B�PJ-l�Gh`i=��Ƽ� ��C��[�vV�m,��-�E!�@�Qv+��Vf=�ڿ����;��3��n���h�#-���q���V�cF�U���t���B���v�!�]��X�40b����c���\�(��@Eޮ�7J��Aq��
��Y�����yo�H�9�ZȌ5~�vLB���Q�"�dm�� �2���XÌQ��8��Y_��/x�y�t������m��;�c����J��)�Ҏ��״�3g��	�����=��O��;g����n���a�9�Y^~s�8��u�B{ ٖ\'��C,,�_>g�t/��3&���s��2��(O�t��\��La��[O�)���J�	
�B���r��T�KcǱ�/H� ��.2c��_��R����G�y�O:)=
p:�a2�l�S w��3G�5���.�~�ܘ>�βZ�9S%��
�B�P(
�?���=r��W8;;;�V�K.�Cz89�l�lǌk9Ƒ�9 ���� ���D��R_5��&X�O`�N�u�9��	9jR�*U�V��Xn2��r�O~y'�+��{�a�i��dW�Q�X���*��_�U �D�P߲�pjIv(`!�ו��ʨ/����pU�m1bF����N)����7�\wu�����P��	�4c6��R�+	�y<_=L��VK��f]����l��B*(ѧ�@snj��e��)J)������H^<	?@�E\��}!S�ۼ�+Ϝ�N��Pf'�v����>%�I��\�z�,�s6����<D�Wy�>�B����/���u$s?�P(Ji�@�JB�!�JhP2����=�NYxْC޴L��ϟHĸ*�JB_�l�ؓ>ׇ�,�t����e�F�����Gt��KlA)���900`U%}��X�q�j�D�/��)�
�O��>Ra؇�J$����L�S`�>�̂�?��f����)���%��sŋ���,��gBr�H*
���Q��;���?�졤Kem�Ј7%��_�!��"�7�P(�}���ǿ�KB�Vl����˘soeV/Q!!�L�2I)EҜx�x~Jف�w���S*�u��������7�T�G��Jiŀ���ߣw���y$�cKn]D��� �@i7��	1�K$��H.y��g�H,"�ٿj}Y<%��+��_\���zt9Q�3��e 1�@�)�_�{W
��Xǵ��?�7��"��vJ���GT*�s���t�@����h_v��R�ߋR� �e���� Vv��B�����)�B��&��y�r�eJ(�A`��\��+��!鐄UJc�1���%�DİXܷ ��X
�p۩_�������Ṁ'7P�'�i�_Fɋpr1_GW�s�:�}%�o�UQ]�7�R����D��)��Ev_
�B�P(�ⴄ��t C�3�@H�?��N��M��@r�:t0�Gr^�)p����x :N�P(�R��&�T+
y�R�� �����K��D$�)���0���uR�˔B[~=U� b�E�A"������*N&�2ԟ�ㇼ�9�g��'>��6J�@�����
9D]	0x�(���W�����+��<�
�B���¯�r�X�p�_˥�U2�?�ޔ?!��H[xsr��?��]Y5'a�p�x[P(
�B�P(��0������;���F��
���RF0P�aAs]�`�b;O� u$�r�^�wΎ���knm�v'��_n@܂�Q_���L"��C�+��"
ED�r=K�:2Ԉ�s`�K1���!mo WS(��
�"Τ(�]Y�����.J�v��o�%��.��~�%��Nٮ��OQE�PJ��xʲTREE  ����������������إ                             4	             �       v                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^d]`E��B�Ez�^B�E�"�KB��` � E�t�Ф�O�JU�Ghҋ(5��ٻ�r�w��s�;;����="Cvbޜ�-9��皶wb�%�-�*gNQ���-۱t�=�Zw����n�*�&A�u��J����e�쾵iA�Lr&���7����Ȯ��E��"�̞>�s&�.\����I�|����Ro�pz�P��V|6v/���������X\n�i2�Sa��L~��9%��M���K�h>U�_,����IB��2sQn��|m�%��b3��͉S����ɹ.�˟dK�e�޴�W$�q�LB�X�F��Q�!���sݧ���ʕ�b5��n�-6T���ڎ�p��{��Y��2U�]x���)������]|�c�Ǒ-��l@�NE*�&%	�&}�㱑*5�vu��=�9˻%GĖ��@/�=ԋ#��̳R�F��Y�D#z�9&�\��=l����{�mfO�h�������Ք��M�1Ro:����Lj�o���'3�=l-:�Ii�����s��^�gU:%[��g�D�rN4�j��Pb`R�P}����n��鰛�kN���_��-��<���(9E��ׯU��0�f���`���e4�=D�)x�����TY��f��[����H5�O��-�����YʋƉ�c�ڢN݁�)��ڂ�d�̕�\!��m���/l���]�
�m�.)#i.���%����z4C8�)WhD�Y��g�ۦ���,hN��x��K�]$\��~|&�#e�7qG��6<�� ����Q(� ��W'��ee��UQĉ�˩�$c�Sަe�N�V��a*Jw� ѝ�D��h�8�| wZ\�e�>�E�����+�%��~]�{pLGU���&��^������s��c����Izp��]�#�2�'����~0e<�ŪĪ���zw����g׺�Ի T��*�>A�%���g���9Z���]h�u˸4�pB�r�`���ZgM��i�wh �u?����	t���y�hV�$/5I������k6���3��G}�����i�
��R	��R��G�k����M���:�"NT�>�����J�&�]�g��B��L��48(66+�5l%��<��~ &T�:(d��1w��݊��m��������)�B�����!�]�g���Y]LY������)]��U�C9�U�ߛ7C����*�aWSN\���佭D�9��+{�^���fQSb#l�R��+���u���v�y��B91[G;��g��N)֒}�R���i�Vd��C�/)��O�N��p���O+���T�� ��{d�7�H��W��م鯢�Æ��I����~®��T�|��912v5hOӧ�_\�I6��DG(�U�+�QWⷰ!~���g:=(��-�bz�AMo�&��By��V�N�B�s=D���Jv#i����ƶg���F7�6�I,w M��(�:)�?t�a���R�	��ô�C��^�]�SV�Y�n�O�I�T#��Y�q+`4�Z$ZKP�)�M�S���O;�a��}��w��(���9���"��7r�Wҽ
�F�[�(1t$�J1y�T�D6a��p�,�k���r�kį�v��������d�@Д!�:s�Fx��9��(ЏZ�)˰�4�7��T�!�Fv����L5׼nA�M�P=9�@ ���������ԒR��u-@�?�Q���-r�J�P��R�!�LV�ܷx@���A�4���st�u�c~T�}����	�������e3a�o�8sW�fG2�v?]�O�7~����n2���yAc��9L�wyװ��xS�1�p�JbBE����Ƴ��a��t1�4��a��Q}Yj��YV*TLD����H��-gȽ������U�F��H�t;VWjБ>n����Z� ���jY�3\����ösZ���:M�T�F��YU���f�.7�Xf�C���1�8z���s]<��<��Z}yj�� ����mҬQ^���B%���]�7���8T�-;7v�?�9G�șt���sM��y5�3����LD�\�e�+�U4'��ul:Y��ѮyÛ%�����[ַ�����_�����
v��r�r
�s]�_œP��/�=�c+>S��n���7C��|����Բ }(�\��@eX&)��\��ד��R��ų���x:�g�a۾��Q���8�����*R+C^���>��Z�_NŅry�OS0nd��\� a-�c�&e�'�h@�l�{ޑu�pL}&�.,�lA03���08�,M�~�C���5мъ���WnN!��M�0�P�o�B�+�#@�܆�I�]I�W��d�<��q=JR
�D�f�LVO7��yP&&�,YP��F�0O�c��BEW"��K;<�c�yɼ�z�䰖�@6P�&�+��}NA�i�-j�D��8��=���'�k^O��j7g�8�J����f��ϕ4�U�A��?�S��GD�lx�G:.5�5��_���غ�Κ�
V8dq��h��}$�Y/� �\�_���['~�TL�π��7V
�
��K�C�\M`A�֫��7�R��C�y�0�����0r���x����`�<K�D�X�??0iqw�!��뮓�n�C�CX}��eW��XxxB�%�a�	��6FsE�f+�?�;��Nܝ.�;X�{���k��r��Hm��_{�/�ؖ�#����u��*\����7:0�
}@V���9�o�eba�����E�f��DP8�V���>�!ӻ�Ɂ��V&{7�Pxnn7%�[�E'�DH�.�S��	��l7�P���#�[Q�.��}��:��� B��:7R/���5�����K�0����s zC���x(�FlפT�=��I�e�d�����0�˽�r=�@$��)�0\M�s��^�����7�I��I�|��xTy�Z^�����՚�r]Wp�u��X��4<e����N��E)Ϫ�q}n�O����FP}�����Z�=>�]2|0$�6R�J~-1�֊F*����cb��[}�k�h=������L����k!�5�z�:���g
�z0��IJ� ��8ڿ��`�ũ��{=z�
�������c�2`���UF�|��}�er�o��n~�d���!t�!���{lN�-�7��ub�q"_�Ȧ������,�MKjJ�,�X��"�x?ZK�jWi�����s5�[i8)&��@��s�����1>n
u#)d�h�_ы\��rC�~�^������I+��{x^HRG����;�k�V����	�R�H�T��pu�Q��Do�>��
��+������z<gw���@j=���%>�J��ۙr�4�g���d�q��mܜ�"�.1���ad�����co$Kb9��U���9��jv˸{c. d��H��aB(
PM0f.�D�E���/�ȉ����,�/z^��&,��`_���r����\WI|�,�&���:�E�V�4�T����k��8�-��8�ꂌ�-�/_BZ�ض�`��\-�F��t�#U�Js(��0m4� �H��\e42�n������Q��9�k��H� \��mH?b��_����j�ku^�q-%F�(�B�õ�/,9%����q�]���=�>�+Mt%�"�6���+f�b���%k�N�)�T��NxsȒ� _[U�1F�.6��H��\�g��	��0�h2�/i�I� �<�U��@ȸ^�Lc:v����9����X�yR2e�4^|�8ϊ�с$aj�s��MR��������C����D!u.&�kl�0�#��E��r[��9��i@9$��%����ۧ�+��ќR��{N�ݏ�D�MSHZ����}���HT<�d��E��"�rO)|��7���v-�.O�� 6�'���R�+��.����h�hM[շ�R��[��`����X��(s�]t2+�_ˉ	�	m���u��6ʖ��ۏFPg�s����7����h�dO����J{$�ಖ�HI�N��N���̉�.��\W-$,�qLކ�U"��"����9S�Z"i'���ޒ�b�&���b2r�L�D�@ErQ�]2[ͪ��^��ˈv���@��:O�X74!d��ا1#)�%8��}Hoo('rT�O�1�����ut�XSܜ��ޠT:^!��q��Rv�����1�Rl��5KDV�.�[
��5��&ܐ������+X�N#��-f�I�ZO�@��ӯ�K���T��jg���p@��*��7�?$�_}�qN��t����hE�<�P�����š�@�W�5)��5��y;ߚBGid��php7�mer}�Av#j�ôp��GҮd�}�0�2w{�o��8rb:ɹ_�R�_k��p
]P���,���i���ќm-9˸��S����=�7
^�\G�;��e8Ձʎg�͎�!\K?���f%��1���^$���uծat�(����>$\3"��A��:K?��f[�����>�8��r��[��6�.��t1-s=֔�][���2��V�E�̛#j�r��`�}E���u�����$����,�W+�OK�w����yT\�nQk1��-�\�_#��p��|튼��ڂڅ`\C
�5���bυ�'��������g)TFGK���_7\�`>W(�����r-|����	���{1�5<����${r�h뾘ZS��@��g\�п�g�p���Yp3@+$ޯ�i<zUS��jL��r%�0Σ�f�T�(��}�qdg,_�T}�:d����P������q���9ߍ�l�n��S�c@xI�L����!st��!RX@�&�B�t]֥b2"rA�nJ�Zv����*�Ҥ���賻izh	?�P��/��>w�,L�x�`��x�!(&�QB��z\S��Í�j�;��FS+�*�1+��ܷв)�!߸X����?����c�M����r��B�E�";�d�	0�����D���w��pRNԼWbl��a��R����.6m0��<��	��:�h��R����mZ&!0��|)P� ��h��,�����H�2N8��̀h��ÙW�v�9=�E0T�-�+,`��Ѓ����5ca@5��S�(�j��u����ش/���'�#ǥ����+�؟�����ǿe���>"�^3��p��y�4Gꇬ��ɉ���=*�_���y+y,/��W�)?u&������j�-�0�~�2����)]�Oa"]B%5�ǵ���*�z�Bd3��7N|��TZl�O�p��N=����2	*�^���և��mwm�G�!���j�����O�ډrb����"�����5���qU?�0�W��n��`�|\�cl�1d>�(��בO�:�<n�_�Ɣ넅�h n*�����/��Ϧ}��
5�Ȇ+��p��Rb�mVN���eR񟫨�b$��h=�
Ĝ���ݩ5B��U���^��x����T0�b*p�]���h���(��ò�Hť�R�� �nA���S�_٨!����[E��Q}��c*Q�_��B����mN��+"ik�ä^V0-U��Ps�g�%71��R�}O��u�	���1�zė�4��ʮgFʁe���X�əM:'����j9����y�z��$�G����<�.%-m�$:Ү$c{9���!��b^�.�����K^�n� �j��)���d� ��MG�����G�GneFIl�(m�Ŵ�=������'�A��M�t�	�ņߑ��1�x\�r���wن2���IZv��qeD�r����-��*�.���!R�O@?nmha�$�ۏ�r����{�+����V:�� k4D���|� ��Eyϧ��X�K��#���A�ej85��D���V���c�ΰ��ým�7i��L�'�t�zT��^�b=�y:7v�ſE�"�4As�S�<M��􎹲	�y�zluo��ş�Y序�Wק+���h�/��# ��q�K����h?�Z���k�5�v|������D:AZ�%�W�8�&�?�k�9hmT��. Jo�nE:	]e![�!�'��I)*4m����(���1>��ke
#����ÿ^j���Zd����@sJ	�xe��k��-y��.��Z�LM��I�/�n׶��+���)���i�Ў�AX�������A�%V�Yu����M?�y�U���6����0���m�Ne����ʷ��p�L�مyӥ وP�O�������3�!��f�P
�ւ�@V_��/�|'��`�S�IWfӎ��uU )Ȇk[�*OR�C)�e��I?�M����nqY�4.����	���յh�\����(Y*[�gw�#����;�0����:���<$v��>�3�`��������|R1����V��c,Y�E�.9&��Ƌܔ�N%p�]�!JqZ�x6����y���m[�^9���v��c�G�x�R�r_iP?�~�%17$���N��N���~@�f-�a�����1�N?%VL��&��а��H�Z5�').}������2��`��w�(Y��`�a��:�+�3r�6����-Me0Mλ-G|���z�!y���d��e ��9��D���Z��^V`j}���[^8�\��j�.��5o�[�$�P��f�l����p��"���ކ�́�,g�l��Y�^�vk��� ��{K	=KW��L�o��M��C�"[���}_�_���(A�n��S�\�l����ˡ�%�l0��sf�B�il�#>]b.i<I[�*��d�-M0sF'q��}���& [(צ�v�K���B�B�6oR�U+{�&��S��!�� is������ز��;��ޙC[��% �ʵ�\4���>�\ш��g���N2zs���VI96k�Y��*S�����f���x�q#*�Y-���Wܛ�\�eěVV����oqA_�t�6!DT��z�j���'!o,JA�o�ľ�yͼ2^Lt���Y	WBo�ВCuNRd�-���q�W�� ��xl�y�_[ ��kl�湉DqZ��	���~��X��^��7D�Q6����չ8�K�[��c:�9 \������*��¿��GU�6`�x\�zsWvG��;[�X�� J�E��J������I,ڦ�e�F��5o�d�[+�D����jm?�M�X{��b
�e"+��
���t�J�k�Ղ��HG2^�J$�Ku���)��-y�ힹ��3���:)n�T��u�H��$�.J�� ����a�r���~M,��?��}�gB�a�܂c����"�t�61 ݽ�Z#[�z����K \	�Ogɩᚶ��������$����=wŧ��ֿ�uTQ���W�"�	�6�&`�gH���tk�!.,1k�~��_�in���&�oW�k�v��xT[��Ǌ�G�r�_���콍�g]�.�!���m[�o��1�1O�wcxV���V��.��]�R���8u_�(�ϵ�`F�Ē�ĳ�X}+ �b�U��k��H�����O_79��Oݦ��y��*�m|�3�o����xOR�R<MX�a���`@�րv�z�'�.a�1������S)f�TxX���S�-���=��+��W�ؕX+��`����U
͉ �\Y�>���}ln�l��[|�v�M>.)��<�}-Ѡ~������b3�R[t(��}����[�<_�}B�%�'`WN"��Ho<��=�̥K�6��C8ļ���Z�X�x�@��7�D�U���e<}K�[��k�􋿦/�GoM�IH4+��l�;%���O|��m�7�@�ԸX/I�s��b!M7�`��;��.4$6��G�,C�8\%����J���T�W�{h��dI�l��]ӄ�n���W��RL8Bi~�@eo3a���,��������ɀ���Y%tΒ�&�ʵ�4��Jx�Z���:4�������aS1�f���H	��9~�QWE����{.:x�ܼU���f� ��$Y���ŔP )�nn=�K$Jܾ���K��d�4͛1�@[�f�C#%j�rU�7"��Y�B|��P���d�8sm�v�k��`�Y38�ч�����șk�uC0�ѱ�Y:{��S��|C�l���d��<듳
�{U}AG3czM݁S^H̵�L�%/*1��|��P�zI�h��������Z�,-�6f��/=Ճ˩cҘ�2F|Ak"��G-�D��uA4�_��H�
m��kp��is�UmitW�V�棖�i%�w�g�hv�Ck��S�jW�-<vR�� ���0��L�n�N���0�e3�\bM���U�c�|BPh����
y�-��ה_w�D�ԋ@H�-�o�(=�T��NP(�N#R����<�+eh���%q�u(%�{�l�@r@}%Rj{��4?��28R���d�,�,�].nABJ�!�5w����ďN�B�MTڥ����[u%u���]�.#�Y]��E�-ń��$YXZ���~Y5*Ɗ!�+����0�G����T���b3�	�~�Y���ު�c��	���^�w6S�T���A�����A:rv�!�[�;	y�L��Eƒe� Ǝ���S�:��
jN�3Z��v��?��*�ef�����SS�L�D^��B&%�#�|��f����,O�=�Ċi5h�c�"��Z�g8E�8�²VX+�kҹV��59��_��=����Z@�6����a��?�[�?��y��15rJ���p:��'�Ⱥ�7:�ޔY��\S:%`��$DP�L��%$�(í��%����-�ߧ��**U��+m
g���������I�L`s��E(֏Jѡ���y���4��wx2�������xm��X�dAn��ޮD�-��R���^�1��P�d��
�O��Am\���,y��M]�
��֫��X34����_ޤ��s]58#�ɡzH}�����LM5k���$P�z�\���2
3k^h,���Z��z#&����,lA�.d!��˗2��5��Z��?�������m�o�#�����j�]CG�J�������%����9�6B��*����V���VX��@M!͊�p}�ܟ���M�΢���Ac���P3kۤ�遺k�j�=��jW�O�,�gh�r�p�_���o��� ��P'��H�~�R[���0C�����<���ݧ^�%%�һ�XӶGc]�(�y�Do������G�5�l%~�T���@�&��_({��h9kĬ��M^�k��N���������K�[�zA�#��u(��Y�t�o8k�����[�ƈ���\�t�|�E|6T+��*ѱ �O�w��s��m�3Ѿ�?��g�0Z�R�@8$\���d9�|�������WG���.0���� ����;	 ���p�g�b����u%N�b1��*&8\>K�J�g�~/`�l��� �@�pc P+�ڑo�	��ƥWP��N�Q�Ѝ�
�`�d�k��ty#��"n�MEj��fn�%:��& '�՝��ȕ+�	�E������:}�q��y��S����sͤ"�Qg����mjyq�J��_����V�s ��)mу#��s���.�����+��;��}a� :��B�%��2�?:�+��:K��5���4���J�E��qf�ù� nak����;�g�㟩s=F�6(��΅S����۱���6ǅ+�,�|�<��P�?:*�5g:�\�ٙk�(�ă�����	+��h�m��F}�=��A�}ӛV�㲣p��p]�i�Tts&���Z>Z�\'��3��+�3�6��5�A��}���7���u"�d�5$�%z����\�O&��ΒLO���35l�㵕�*�
M���V�+�)B�RP�]V
�n)(�µ���	}m�U��-t��o~0�ʃ6��^�۪��U�� ���A�6S�ꮺxy��?�3�R��,�w��,���E�O��_�*���E8J�z�66#Ґ5H4���Gw�Rꯀ��d����)0�;����r������x�R:�.2G��Tj
,�-�ء�{V��Z=� ��#�{� E�=���K>ٶֺ�Q��VF�O�h�X��s�a�Ȭ���󩾟��G�C�i�wܦ��ܐu�'H���,�Zv�˃'���#wS�[�.�Y7ѠT�s��wa�j���M�ʮ�֛�K�n�9���g]����>ȗ�L	.(j,�6�H�}-´	J������ǀ�a%o�����@� �d�xp�U���g�T-.�:�3��׉Il�0g� ��py���1J��*�,�W��o�W�T��Kq	CvQ�P��U��?��P�����(��A����j�%i���|����A�7�,�ۭ~|i��ԧ�N���EK��kNw���%(ȕ%�>��u�� mt�|�`
`_�p�n����)+\�I���\s���{,4A�Z:���|���uk�Jd�ҷ����W����zVH��w~ׅ�g6\O�Ҡ����	���0��*S�t�\��F�Y5d7��\�$�ǆ���#�k�/1`��7��>���e�B4��AS����1;W��Bt�"�
dc|L�V����;��
Y�_�Bh�#تs���%����zlÍf脕���d��2)�y��1��R�7u���H7C����{0µg�����~S��m7P��������u/טҴ��XCX�S[`��9d�r�npn�	z���;�sX���:p��bgF/����e���_@���bL4�ֶWn�������<Z�)k0iV.�f��A�(@v&i�A]|)�?`�۷�kD�������ע�p"�c��+�Ԋu�,ŀ뷙+L�����K�	���ۺ��	�!����OL�f��|��� '���%\��u��TE�nıp]}�n�+��OT+
�X�U�
bqu�1u[,\ג�20jk���q͢��ZD��ñp=`Z�F��K�Ѝ��X�fg�\;?�{FA���;㬷q��$|[�U�^�)\E�	ײ.��~�5�p�F@l�U��_C�+5mĒ4�|)����߽��5��ZW�>+�]=��dV���M�_���o�7I�f�s@30�I3���iQ>i�=}�+�e�Ւ�Y�f���a�V�v�BC2�)�u��|p%�\��iӉb`��6�>^WY�r�Y�u�X��fk䛼�˄��Cs�+��g�[-c)�?k-��*����k{󬝫��C�D���\q���������x\��黹����&<J�+.x�\
WJ��������V��6Q&���puf�_L-`���>�\�hg��3W�+ד-9�b�\}�|��$�x�CO�+@��6��w��BV8�D�&G��7H��,=�F���B9
�FV���Q�V޳<�sȰp-՜�b���p=H�t���4ݡ���q���_�
e<��a��X��?KS����d�Y�Y�?�7��f$у4��	D2h�UM�&b�,H�ډ��� �}��z~��as�l�W6���������AG��aalN��a�:���µ��`r0�vP^�kz����	�\�K���$z�y�S�=��r���\_���K{zݲ�
כ�+�	��s��޳NU�������}�kJ]�k��V�Ǖk��<�1�<X��}��^4e�e��\�=��Zk	qD���q�v� ]HG�][a	���Y��up��y�q�M%�t�n�8�_>�$l� �R=�[�GЭg��pX��=[8+_1�f��
�ʵ��6?kq�ǅ��7H�..�_�p��OrPi t�(�Ԥ�ce�c�.���,����	����2v�/��9fNs�u.��M�OU9���u%/�)����h��L3@���l3*�~*;��E�ĮJ� ���v�Xᫀ�[�*���kJh:~��rm���?���� D�9(Cl�T !��`��7��a��8N�HB	����=�:����p��yk��3W��G�������ͨ�y���˴�7��x��%ͼc���u<"7.�Nn���Dd)BKں�
=�#}7���zݡ�0ِr]�_>ƚ�[�3B�4 ��D�g�r�PƑ�KH�K�M�3]�z��˵���G�f��SD&R��u���������Xɞ��*�����h�:��#���2Q����KMR"�n�\Y��2Z�~Oi����r2'�+���2X��W���gו�s���\Sb-\p��K�uK��/��:r�*�k�]E�FsG���@"����T�$\���+��o��h��~�r�w���"[������8t��o�xiě��fǨ�D���S�}/���+Ǭ�޼H7[c��7�xT��G�c�~��q�fU�^*���5�S�qV-�^���x~;U۟����3(W���w����ذ�b��k��0yy�<�h�E�N������ ����<Gl���Kci�6iU�i�YWyIZ�5.	�f�"�Gx��A��_s	C���[����|ߚ$�?Kk�ʖ&��>r߰G,@������V�B���Rk��:+�#[� �5��nj�d{���i�`��͈u?�u�����D]7��)c)F�4ͮH�Ji�33��x$b?,ٟW�/l{�6!v5�~�\��i[��l���4���5��_���\Z��Ԯ�WY+�"[�����¡�(�H>���m&5�^�����µ�.�)������p���p��������Am���@�H�En�QTZ�q���!�\�l��(j��//�k�t20��	l���.��塱�jaӐ��>s�f>1G��,�~CE����6�b�@ZP؞�����\�+3e��v��M[��M[ݸqW�k#2!�7���ɶ� Zr��X�4UCՎ�|�Qdx�_R�?��a�����KP���i���2��w�M�i�ix���@�q���A�Bu7�-cy2w!`��P��%���dF΃�^�d��A����ǿ�`���3cp����"u7A�u��h����
���W<��kI�ҽ�*���}sLH����������P��S�;#�e?����I�54��a����Ty�Q�b�:#�]���bѧ$��m��F�T/צ��r�_a��B�˪$_+3��0>X��?���i�[$V��U�Ŷ5�	{	���ЖŚڤȵ��h��4��	b�BD����d��)7��"�ּM	"���-��
���[�~�C t�]X���,�{x�G��y�SF"���ΞO"h	yՅ��
�����܉M�Z��V�>W�v-�x�rW(
��u�h�c��jD�Y�Bo����A�U�OtT���z��̔��@�*Ү��ez�dqtq�J���Ƞ;KiC����Z�ܜ5�[���ɚ H��b�)~����;K7uê��׈7�@�	��k��u�m�!�.WE���ޥ���<�~_�o ��l�>@M��#<��6a�š������J/*s�=�D`A���	5F��;�Dhآ*$�7w�FcZߟ	6�G.K�%��PG���b~[� KH�0h����r-���Yf�&{k�Z�!�uE�As��l�̩�_�[��*��!�fd_5����h�zԜ�7eV-m�
>ER%>04��7^@I�%[��O->��B4j��th�.>�����@����?,q_�H|�ڇ-ӛC��2��.����=���e`������p�:�E���p����sK+��_��jm*�|uJK��UȲza������qO�s��<���^�o)�<���o�����bv.J��j��b���HQ��"��x$�2�|!Ͽξ���ϗ��9"� �ZD ��Cf����kO�`��c��*�O*����S�U�[�ì�Y��d�%�D�!)�h�'��&阒Bﱋ�����j�ȧuG=q\쾗ʇ�S �Q]sgt���*�ES�t�#T^
�]%�(�@J�s�,�04��߭b�^����D�]z����o�q�VMG~l
\����'OX�*a�	Î`���<�kt���duv����
����2�T%Y<�����=q����HZ����� _�<�b3ц��>G��ȑ���\~����r!��u�������/X�jopt(���������v�SJ��5�Y�kط��t{����5��w�`D��P�E�=�+�R�l9)KI����K�iKw�ː:�9�r"뇍b)�A۪~+_p*��9}8#s�)_Rb��N1����>%\�Z���x���A�BT7M������&c�����:�G��>��泴\�/��^���ģ���z���5p�>Oy�<��N*\���e��3���1C� �Ǖv&��o(44��]bk=������熨բ�������'d��$��b ��9��4��A�9z�NNS�V�\M�_�ԡ���b��nN��GFf��X����,��e丆x>�
c�T�P���<�P�IS��k��@+��@����0r���׫���L<�����"�~�(W\W�්�)(%5�h"�fZta�8d0`%'wox"��έ��7Д���=�эK��KA%�q�p�(��06 TV�p�H�u���������A?E��:�%�4�^����j^�o��l A��|}�R��]3��6p��R#�L�����*n��b��
�(|b�V�9� ����9񟮑�G��eA����խA9KT�	����MpE�)�=xр��`��+^<͔S>������a��7/-������g'}��C]t
���B<�,_���؆��ޅ�J�(L��FR��,/� pGlP2�����u�|:�'&��M����CY���
9-c��"{)�����.ê�ݬO�B��cEd5ꔾF*���G��8s]Ūњ/7�@��֟���n���A�+��k�������׿t���+�eMs�8�
F4��sS �p�����h�ޱ��G�C�C�#GD�G҉��<����I<���-E+�I���	�,�y#�4��&9h�[��$��C��D�7S���əb�H� �e����@w��/$z���?hB>�3<Η��j�3'��^��Zc�_PR���&8 ��"�͞�&č�X��N@���&�\^#M�A�� :��B+o�w	���d��Q�j�ɗ���S�b�Bx��9�L�8����ͳ����O e�R����۫�D�"��Q���H=3�s�a�hvՂ���7ex�q6��$�/�!GxuZ��8��]���((L�g\G}��r�C7����t[��yf�P95s�A����s=���'GLS�pʐ�R���I/���z��,|r\�9��y���i��[�82]���r��S��������dw��C��'�b�<��I�1KVĔ�m1M;吇��%�`U7����Y9��0w�,1{6���ۀ�s��9� ������SZ��Bͽ7��W���<��͞:���3����_j8�y9���*�i����kr�=>\�YQ^]�g�^ߺe�����F�Õ�#�C@���\F�Z]Me�;K;I[C��ʵ���!��N8��:���rԃ z�mJr$�QR2�?4DV���ǵ6��Q��Q��"��Tݲ�
�����Bi�P���}VǕ���MK���H
�C�E�t�������^��|������?A*���:gÛ���xK��bqϫq��8�43�.l˝�*�E2R��ܮ�+-C���9vsf�2���J_f1�f��V�Z�>�����'��V�4@}sL5�k4��G��aF��������\;��J1��<<�Hx2���ϻ�S�ڛ�/��z�~	��AB�u�.Z���j}���kB� ��|izwUiq'9�V���oq!,aj/�̜��(^˙��y�x����� �+ ��7��nE4�mMx?�c��"ց���U��,���ၜ;Í~*�}~A)5d�g���^ͩ"݋G��D�Qc�쇽M+��Wy�D���Q
u���n�M�>ʖX2. �-��X�q-6n�4�HbT�~Mke�Dì[���\���T�C��Q�s�q]���G]0ʲb��Ie������M	��XN�T�?��WSA� Ζd�G�7F���<w��c���_�E�}���v��JeU���6\��82�Ҁ��R�u�g��z��;�_� Y⾿��ΉI/�������4�$��6�Kũ��U���p][)�f����6��*�ԯSJ�<6�����(j5m�2r;YL'��K"{`��9D_�`����~��](C�k��.[D�p`�!ܭu D�0S�����'�VkA�����#%f9��/�UX����ʭ��kP�)�xp��"Mi�Ѓ ��г�#T���X{&��Zh��,C���}�_�7 ���
qN~�\��©��tX��iB�F~�ch��W�ʂ1�}�M��kQ�k9�X���5���4_b8[d�MD��B��"v֚�9~ګ��=9Z�v���U9�]�_���1?�)k�d~x�ǵ-��Rԣ��&G"͡?�Ũ�x��N��Ge�2⌀��ީ��|B����%���E<�Gr��a�G{\�`Ȳ��\T�+��2â�OtU�=R2��W�nU`���k��ꓓD5�=�J/�.���V(�����(S�-Doiر%'�����pE��ort���T�n������D�c�R�~��ʨ����p}?6�t���p"�Hz\� �@ʦ�\�׌����������2gtM�S=2\������l�H�3xaM��ZE(�8��bd��Xv�ݓh&�i�µ ��կce�#��uƶ�jt�%�!ƍ�Cǡ�Ꟁ��N�B�j��\#��9/Iؓ_�@�+�� jv5�З�?�3����_�RC�R=���5�'��(9��W�x�p�#�[Kk+֐3�����Ώ��O<��냢�ߺu������즖�V́be;1M�7�ʚ�1�/I�T�eogu�5���L�3x
���ýk�!�"}z��֍��:;��aG����HZ·�8�I'��Ǖ����c�z����Ala`z�p�BZ)x�/9����~�贙۸O�����V:�^�ul��$B�+�~�4�e���z�+�������kA��#!�k���Ep�Cy�?G�c�X��~�[��.q:P��?�pA��!�J�Q�'���<H�Tr��)����YGzg�OT�����q����^�Լ)����z�(ʐ�7��H�P[9'�c�҂/���g��ǵ���'��w�8�"��Wg9�������sj!?�g��Dc7��~��"��_s4o��Յ�,(�YB�4�p�����Jj�I�[��+���_,A�h��Z ����m����t��kr��&��R�:h�7/p�"�k�W�HXtý����y�(�=����u������:$%z�\)���#彑�U�DC��Ei@��i��~ާ����8:6�@�q+ؼ9ir��~(��[�"F�����o ��e�K���,Q袕�VR	�iW�ǳ��w��=��2$�8c�v+����$F.m<�i�֪�g�a*�#�����7N�ב�k�ڶ��1�V?�Ī;g����a*ڧU|����i�|�k��v�/M����u|}r���v�|��(7�c�	�Y?��\3\|`~������U�{���S�ޗ�Cq[��=��U�Ŝ��lx�~3��s�n��A�P�,e��+���)���#75}2�����1>����I�³3:^K��RHn&�kҀ��MU��p5U^)�{�Zt =z�Y���|���ٙz� ��:R"������?`�(�.$k�G�����]��_:�g��y4\�8���o
��aB��@\�\��ܵc+�3�gs;iQPO�{�L� �CD�6s�nD"���,����?v-Jm#��D��[���?����/��I���Ul;���x�H��a��m�[y�����^��i�o:������Z���4:�,1��J{5��
���1��*�Q�V�ˆ�-��5���q���nrX:�� Ϻ��b��;d~�+���� F��B�@t}�X�g��p���զ��������y��OЄ9���*Ώ���y_�v~�3.�6���
$ ��c����l�fT��	1s��u�7��	CɃo���"B��}��� '���Ѥ�Ȗ��U]I�D����\V����;���nN{7�輪�xX W���^N���G������������H3(�;ψ�s��v�c���Cr2+2�n��%�M��U��8㰥r��R-7���yϊ�3�:��>�֎Y;�� -+u�=�|��()�is�����a�g�iޠ��H�R��_z�>�#N)��:H؎�_M���}ړ��� �1*�:y�����	X�����㨥s��������ݐ�4#l7"�����i]ǥ]Z'�s@j���"�%�I�-c�<��b'�>+�-���9f5�>3�P��r�s7~��&�U�1�XC����Ma��������B/J����k1͊��S�!襅'�ؿ2�r�968̽�[|�������X�fz����U!��d��ʰ�'�T��Y~+�F�Ę���u���cw��Lʟ�<��7�<�X�d,fӡp)Ǖ5P�ȋ!�k&3�	���I{뗢AC��h	W口s�Gv蕜cm?@���4-�n�}��q}Hw.w�Y�z�-%� F~�?0�B�jӜ����f4mY��}�T�T��1�%G*_u��A4�9�f�k_G�YN]3>��l�i^�2����t���S�Y�:NO��n!� ]M�/�X8N}2kI$���h2n��3@�AT�PZ�OT2��RJ�E�:�N8�)D<��h"Vl�S��=�96�?�YB�\��C�p�LQn�8c���_�ñ�h�H�!�!��s�z�D�b������v$�q�z��9����$:���3)L/�3T�����U)�9r�Ʀ��i��&��BS�D�8�bjOL�l���뉞)�qT��$����L��(�)�*�}����t&)�	}�p��G�����ņC�=r�g���Kh���1��Q:b�\�����h�`�`R*�������L� AШ�/M���/ٸ�J��݋n����m�����G@1�:Kђ�䱥�)3�r�0թe�5/@��Dr���=u�yFe����˻4Կ/9�O��uo�h3;�Y����`�t������6uIC1���ޕS��d���o�=	�'�ŵ�T��pT���(�B)��,������c��шX�k�MW|��'��t��������U�.5��%=�X�:����k���\��{�곇xe���Ք��	/׵s�ײy�e�u�d�b*����R4�w��j����2��	��rm]�E���棜h��
��Lg�\y���.�G��z�<'e���B�`r[��k�:�9ׅk]j��/����u�k�����sdQ�t6j�%�Z���A�Ɗ	醈_c9�_;Ԗp��h��һ?�)�
���k��j�_�}��U�ا':�u� (�=��a㹓NLHg��Ϻr���\��F7�C�f�!Nhb�S���S?cO<�7b'L4`��u��]��7�q0^�i�R ,?u5��6MG���T��ݫu@�q���!��doez��6�/F��8�ϣ
@��Zj�|&��YJЋZ&�Ex�	���,?��/�`���t 䕸�P>K�HnO�r���I��v��u5M��8q�����wh��2���@$f�:�'�Q��v�j|�e�d�Z��\����|E�?>�a�ցA�a���՜�8�0o�웮�ܒQ�L֕�ԫ����knW��ߋg2��)���
3�K֯�B�>�J|:��ٞ+.��/?n�⩶�C-��4�h&Hbj�P����O"ˁ=m�P�5PŽunx�.�WG��*֣l$�x��e�S�L��O����3G�:eݛw8�����-!��19h4�U|tNI[�
�^?j4���6�5z�_
QW�~��ts���A�ȹ�س��FV�#�v�N�"��.���{\[��ٮ�y�=����Q?U��u��8͟x.�m���L�����+��2?̃����^7Ӛ쬪��'Xy��4\�p$�0w��6ON\\eR��!V	ϟ��s�e0s���e!+��*��b�#>�7-N�ƝH���>������ݡT��d�s6�:Z��Q��[�ҵiH��C9`��$뛘R;�ջ���N�YP����9�6d0ײ���%]�6x͖FN����lYTMڌ�@R��(m���N��5�q�I-�3PK9��t�,!-.���N�ͬW���%N��&CT��k/���P.7�K���H��.'��&������	LKr����E���B�N���&��#N���_/h�g~�<g���_��b	�cHr�z/�:�p�ak����U	�=���k�p@Yhi����:�%`�6*��͌�]§����w��[f��3Aqt�}�h��F���KW��o)�?[�_ ��'l⎰ Z�}?�I?I�3�2��,%�0��D�&�h&ba�PNN�CO�Ty<N��-�=�KN���F�b< Gf}�Ƥ7K	�����skC�Db�c^��$��� t~�H9�3�i�$,�#�7R�-�5�1ED"����,�7�J�a�VOjٚ���Db�t� r0C��1!4zo�;�\�3dQ�$��^�Ѣ9<�/�K@��D1����SΣ3��|���p> H-)���O�P���V��-;��+Õ2�H��WJ���8��u�$���R$U�=��-<Ϳ"�(`Q|k�:���W��*�Rj9YV�*���F��s޸�ք����(�C+�|��l��8\Uk;F�Yu�T��@�%޳N��ыP���_37��Z�D���� ��Ԫ-�8��c���Nѹ�$�ӛZu��рwˈ	�E�ɅN$��B�'r�4@M�L���mG-B�a���M��d�*rx\��_]�$�~�S����k��s��b��"r�k���_�8�6B����#���t�|Cu 5�Q5=s�sN#��5��}�T7+�CP}�y3x��q��nEj�f���4�w�<ϣ[�o_�G�
Z��ى�<?a|�3C���Um�xlއ�?݄&���u��Uf)z�#�Z�ӯ�ˁT��2��"jof_��0L�����
�p�K�s��Z��[������$ޤ���/�d�����'�{�֕l��\깥��gv��k���Wvheb�"�f_H�<ds��Mrr�P�F��Ε�w�Lm�I���K�i��J^빔����I^1D���/̋�W<I���<o�����$�nV=���P�`$�l�Z&>��ޚ���;m͛��,0!뀥�����o|?c�qz�.�o�kD����Ϫ����3-�2uts���� ̗�p��:N�;R�����OE�K�=�s��ϩ��y╁�)TU>b�,]0͚��[�*�<?}�rhA���������^�i�u���9��L�F�yh^Gj:1�7��1����>iL�x��Y���q$��<[�=$�z~}��cH�5�<���I�`��P:=���jI�<�t�4M�&* ���C�5@�FD��q4&�z+����]A�|i�e��C=�(��v���?Ņl�<@��E'�����̫��6�)�Q��M��,
��UkA�O"���r^�h	��fO	�!@���:�1:��K+A�ǡU�o�H�m�����W���5_.i����csC�'���_�Ϝ �߅Z2���j�$'���əp~���~��(�"��M�@{'I/��O`��oL9Ni9�*,�Z�(
y��?�V�8#*�r�^� �mh��!�79ɲUX�DNF��v�֭9f!b�9�-PM̤���@�LO��e�9��	�3B"���V��s����.���N@�"�z%L��e���[�z�-�+!鿑;�p
��������r���8�V�]yo����!L��3��ZWvٲל�Tr*#ZF
�!/?�4��S����8��}�YV
�����������q�`Z��[��3�����Kt��h�H	��*h�p��s�9 '��*-��3��W˛�s��?M��*�Д/D+��|8+�ڰ�k���늜���0DC�m�mG�mc�V1���HL ��L���j��JG�GuT
�r�;�|$3�� |T�{T����r�ʙ�������a�	/�s�G��Z���>�ڞ�����e�HF�l_��3�RΜ�=k3���t����dlO�T�ӅM� l�N�R}#'�������c�\rf�_W�-����%F@����/68��Sc��\����ؽ���OӮ���'��������𽆈^��3+�5z�߽��҈O' W.�-��Y�jn@�ESl�<1���]85e%�H��V�*\�.O$�8�+�����^�y^��cPˢ�/�5q��WS\��rD[LiV+l2�oq��=sd9�bs-�'hN��ψ��j��g���X��,���`�m_jAS�%6��2����?[P�Ly�k�vu���u�WQ<�!@�*%�EBP:H)���!���n�E��Жb(Az��#"�B/1(U������{��|�������now���P�4��x��:��<:3/>`�v�o4�K@-���bڇW.��*Cw�*�!����k��T_R�Y��
���57�ȃ+��τ��q7��8�!���dt�I��U���PB�<b�z��;��"D>9����Mv����zPg�P�m�`���/����C�(��	���~V�x�Sy2�яIt;��h��7���ܯl�P�5��ɠ�P�

6).���9�zֶ�D]���W_Oa|(-�g�R���&�d(-̙Q�~s:
{:��-s�=��U�)vV��ԥ��D��B',Օt����ғ�|��q�ڔ�,��{.��R埮N���ꄍttB��
��J��r���:���D�t;��8�`�Gе�|�����'
j_���!yQJ���B4n_���[6;�붃���U�Zđ8�q�6m��򽦹C+�<G���fQԛgci� <� ͺEPu��j뼱��zkI����.��������)��J�J����ٻ��F���V��y@��hSԺ�O��z�r�ma�}�nӢ.�v~i�>[;.�H�t��	�����Q����~I�Q����;?�skYڨ/	C�(�����?$�Y*[��Lfyl���2AOP��CO�x^����у��m:�b�Cgt(�'٤.W�1����0P�P��֤�&&�o��e��z���yk)mֹ[�nl��//K��?K뾳��8�&�fi|��W)�PY,�.�Yt�2Rwh��iC�EI3���/�Vt�/���~p�e�#��Zp���7�~�sg��|0'p����Q��ȥ�wk�u�\O�ﳐ�����Sɚ"+l��j��H��v��tG2���6�,����8�}L�#>�S�?�#��!��	�v�e+��n ��i3*H��b���>�N�ԫ��9x~�9�L��'�$42�d���%�����p��G�ѩCngv�Ǟ��`0p��%�f��~j��ZS�=�Ǧ�������`J��� 5׺���o�s�ك�ks���V|��	`��j_����ŅF�Dia�6��+�RNE<��a*f]��5�!�F��sAq�ZJ�Ƈ���uOVyZ�إ�K��|6�m�]�DWq騸l�����w(ݬ筥tY�:���,���e�2�;��"ye��V����}qA�E(6+I�_��~��ߎ��RZ���k�،�9ğj��Jel���Y�{*�U8h}2��wR���q�t���8�"x�u�e�#��^v�-��p7W��!_O������-���0^v�`��!��7�D�¾�;��0�?kS0�\��v��W{!�A=�CK���-]7P�,��1S���y�>v)��r�`��l��P��[~cN��RhU�ڥ����lN���SY��j���"f�0�p��Vɩqf<L�Q�4WU���>�|��]-�:N=��ǁzN.Bs���݃�t@|W2k���E�Qp�G�/��7'�kZ�su�LYbwJ��X��G��K�M<#+z�:G��T��K>=f�<�{V�m��U� ��z�X���~tm
'j*�/f��L�O��3��	q��CU��Ɍ~�Eb3S��܍Лڐ_*r�s{X���+�z�O�w�B������DYV?�>8��|Z~��Ѽ�Zĉ��4�?v!oG�f���~�"��?���~�Wgd��b�!���ȏ�I�ik,�ɏ��@}�S�
)w��"���C��'�jgl��V�|��.Zޫ!�~8F�)�@M�G=������Չln�~�n}D�Ŭ��+���B�m�U
��?�+h������g�n��z�b+�DM�g�?C��V6���@Be�D�W���g��P�Xj��b}��Lը(v{��yfa�v�lq�J�Q��g�º���:�:"������8�Ry.��G����;BɊ��
��rw�G�;Ŗ�\�'�G<�dqSS�E��9����{�Kx8�#�e9�I�N>�p��D}՟� ���e�](r��ڙQ�w����!%��Go	�Mq�r�E���cʾ�^-A�*ZH�:OI��"a�qZ��SE¥�u�^Q\���j�0j�e�s)Ŷ
������J��"����X���{�-���(��3Ϗ�	;%z����<^˷�q�E�8��<��Nvv� 
�����m_����Բ�X[UDR=Ts��W֖�J�=2A���^�i�~g�c���t�t�E�r\��$�EwQ�b��b6���L˪׳�F'}�.p�mI2W#�슈�)6+T�
�~Y��>Մ~ѕ)a$��}qT�������A�|��Wݰf���n��Q������vu�筥�Py��$�����h�e��Z��-V>�j��r��1�f��%�_�G������0�S�9�U������)0�y�6���k�/jd-)�����]�m�)"4k��j����~p�x��!��=m�w��G�$�,|����K���6�����Q�F�!���ڤ�����)j-è�U�JAX�3�#��J_t٣�{�G^��$���,�<*��[&j��&(S��X��_�|+���\Ñ9"^7�ϞD���Z�}�i�
��+�'��G�/ҹ���MT�����Y��a�������ʒk.*�P]Q$m���ߠ�F��{?�[K�sYgm���ᔳĵ�ӕp��E�(L�
�^�E��,���ۑ�7�%��Iq�Z��/���,Q��Bz*J�9��E%L�ʝէO���G���K�U`ME���&�{��'Fwuj}J��T��N���YI���FXԌ�{��ӂFs�>��A)��߂��rD����Lۛ�k/������ь����ſ�Z�ޔkK�K�6�<�pQ�/���(��!���@�ȁk��R����te�����Ä�'~�Z���-��,ْ!?C���0+�=;��Π\������`]p%�cD��Au�0����k�w�t��
(�
I��X'����^�rϼwoL^�����rc ���s��B��[@�i�Lc�g� z�kz��T�ʼ�71+i�yn�0�ڊs3�c�S�|���E3�T�*�)��c�.o�p��H� ,����m��yWQ�8��wuVx���7J��]G��+�4�=\U.?��[q׿�i��V���J�y��և�_�_5q>Z��[P�(8���M��8�:�]4j��7EJP���Y�R��R�nA���}�����f�UkG�R��O(�Դ�+P+Oԙ�-�-l�j��yQ�v`1뢲�/����#f<v'/��m�7	*ی�IjJu\?A������]d砎"�Ӆ�m�2?�+*rv��a-3w�l��~�^רe�~�-,���zx�����ZS�?~�Wo�����<���f���	F�;������a'b޴� -�_������ծ�h������T�%~�gx�O(���a�{��'l�x��1���#�7�	��9׉��O�>�".j�;�44���|A�XLA���k�����0��]��௚IPg��=�F-�dkc��a�x<���:H��/A-2j�����w]��t7��jrN���+�z��e��RA����zX� ꕈ��G1^��F�@��uD�=��^U��a��v��ћh�a�N}�&���JQGt}!�=��C�
�aA�J���nFo�Z��qP3������èY���_���A�R2_��'4@W@m)��ߞ>M6�½jP��1���`*�,*�7AA���I P�桅Q�B��9@]˿kb"��A��3=�ڂڛ�kC��S���
���[��Lp�UCk�������w.2����ʕ=�/�FMK������l��G�r*�U����������Am�j��+��(j3��}{��}�<�Z$��of{<���T�˫dT�����o��EM���+��^�n�������U���2#<�2M�oǓ�'<b��Y��Z�q�Gt���w׋�v&ف8xk�)�*�_�h"��{}��AE|�5A����3~4���Q�붨������~�u�F�Ӎq�G��fɃ��
��R�ن.��笜OИn6'����4(峛�ST$����-��D#��呯cţ���D���y��st�UtJ|e'q����2�]uK'�9?��- �[l�>��A��re8�~I2�R���!+��D�ӑ�Z@؋����q����+�¢�T��*�M./G0�-))��2��&�'�-T��Bc�m�l���nHQGzg5�g�d�$*0G2ޓ�V�45к�g�ǶƶV��� c- ��� П�hM�/�:��?�K�?�۠�Z�]��A�9��a"��}��)P�^$�,=�d������|�:Cag�8�}����k��_�� g} �-� �x=�f�T��<���#�0b~�̀��b y0�BT������-����)F��4�����tApx481u����g�|Y�s&���]N�k�T�T����:���:��ـ���ߥ5(K�aF-=�4�&�-�?ԣJ��N��;f�+�?̍|��s*��������ܿ!�;�Ɂ�?n�'��Vh40�m���%�<�GE_߾�F%��*��w��AWՊ��Fl���>��T�}�,��$Y,g3D��n��Rמ݌��Trg�j+ߗY�����¹�jF���B�+�Py)Tڻ��!V��{��q�Q�v�=����e�.�vR��)i��[${��K�����'s��c$�@{|QA�/�Q��y��=�B��AM\֌z���`UPE�)qV��7[��IN㉠i✊΁���r̼Fg0P8kV����\O��&���}63z����#�s��/�x����e۱���e}�2��H��oE�s�ѽ��LX1m,�Wl�ڑJ�ז��pm����	֛�S�m�%=�Tg-���}:.����r6^�4fOo*���6dטa���Q����j(q
]�+���q�lm�pD�?���,��0�*���BbR���?Lo��RM����E�=,�����v;�'�!��
N%�{g`}��9S{�Ǘ���%q!�Y!�K��$C4�x�k��B��<��I��/�)���n+�ZK&_���/��J�*��8�����}Vz���.�t�ԟ �LSzk�(�����l�;�u�w~[��7�J�R�y�,j�AL�;��#�+'��Ż�h�F�7E�uIP��5a�5*!��Πf95�!�>����/yH{�eFuuޠ�2�և �2��P�Lu�ޡ<�+��m��0W�x�Lt��&ՙ]�o�6�@hb�i��<Wd��ea�Y+�m�E	�I���#w BJ�����C��{�i�a����t�ٵo~��D6��XU���?	*���;y���eg�	A�?�7A� ;�,�m+���p��m_��༏�iP	?f;�{�9�x�?	��@�_U�ǫH���g��`	Ŏ�ݥ��{����!�򱍖y(�#]�����So��mP����,�����u��.�����R��J�_� �U��w����t�;٥T{�+���J� ｒ�J����0o =�=u\��͝wY�y,Upw�F[i�&�(U�4�	�<b���W�s)��1mo�}�jD��~&�Ny�0%���;`r:�Wm�ɣZ4A1'�K����w�9���Y�j�zx��7zm�&U~s���ճ��۲�k�[�$V��J���?#XJnY�W֙�����ߒG:����}�	�9J~�u�*`2^�>NQ�.�LN��>َ�>%�3�x�=�X��3�xI1]�"�h����7ǒ�W�)r}]���Px�e7qX�Rє�yl�2�0�j2֬��7+J��Gu~Hu.l�:b#�z�9�sV?�x�-�m�����1&~��Ё�|(|�Q,����)�;+�s9#�Pd-Sk��t��*
~oU���>x�v!���Xh�,�֜݉�F�ʑ��rĻ�;]��<[�E�齯�l�aO��ٓ����l��M'<Хw�]j4r�<�]a<�t��ǟ����@��bd�iwgw��y��L�PmQK�Kh���w���[�-N��� �}!-i�q�o=�I]=�^U1@_�K!���%L@:�X�r��ˎ�W��7:9-�N��Y�b�Y�u�X�����J��O��g�#�]eW���MX�lQ����Θ5Lպ� D#��]��������Y,|;�G��>?JY��:4�pofj��si�R���N@�������WX������}��_��*�=�~\��^bq�׳N"���W�_�Q�2��s(o���9�\��?t>��?��{��$���ɨX�]� �/"x���0��I�/T���
�7��К�zAg�=��U����P9��l�h����	n�qhE�8�]�Y��JvR⿢]5�C�4�j;����GO��:ũO�S�^���x.�6�3Oy6*Re��S~4 �i�� h�����}�ڰE�K�*-�Am6,q�>)��G����d��+2c���	�@i��Ud4�x�ַ�{J%A��G%��o���e�G9��d��]�uze�9�M~���ݎ�k~��-1ͷ3)��m{I�V���3,������-����T�_�ć��,�g�om9&�.�\J��\��e�P�'�i�Q�ʻ�9ST���xT�8r9/����gw_�A�R;���*S�7��7þ��/�,�Tկ��j�^ʡ_�yH^G}�.�b��v�ȡ?�/��$��T�c��[���'uf��8�����J�����>k+��Ic�����oʔ����7U~�{A���;�����,g�-|�c΍N7�Vǡ��0��|Do
۴	��
���7���#��kQO\�m�����)����j�z<@�ܽ��e��>��&�����ދSYmYS���(���*�����Ъ�vw_:?��/�A�7�3@
]O���>�@���)m~еΦS���L�|q����D7
��C��0n�Ml�O-.e�P2���q9�[}m�!��Zx��t��W�Ծ�A~��Y�B������B�^��ޏ�x�g�'����ݩ�az��×�'�f����D���t���g�<�����j�* i����A�S�T���P?j����@���Z�(妩x;�<����C��d���b���a��I�ކN�i�,�sh���z�>tK��Ux�Gǋ�E�#�ۿ��#Q2P����s���Ȱq��w�I�.V���r�,����l�N�a$�B9��Q�&H�c"��-�8��
�v�o�2�UZ�on�8�J��!C�`����.��&����5%g���C�	%�D�f��.'wW�.�>������[M��q�ߘ�r޿�Ѳ#Hd��Wh4؈zG�+�jx-L6�i�%b�9vMz�D�qI(�r$��	NY�mlD�
2�]9�	���� � :(D�>����i�kN�3��a�����8Е��ж������I�S��u#�vm9)��ا7�!����$�)�_ڜ�>L�6{_o�Bv�O҇m��G��ח�0(�$�����<2��w��~��>*>遬�QJ[S?��i&?Y^��^�m[9��ˡ�Yřw�:��7k����>�; ��<�=j�E�|[Xh��vФ�{-좺-�*��C~��ރqށ`L9�d��:Jr6�?�5��"��ע��o5q6>qqc�{�,���*
��}z���c��O�W��^��m��H�K��@3Q�J�wZ!�l�d �t2�y����E�(˾�)�7�Օ���E:��u�t����H�u�K��#5���>��22��,�%���m&�F#��"S��v�5�'#���	�v8c2{0���O�%<?���H'��5pN2}� l]��n�./Al�K�qv�'v�c�]�?���,���i�/�$&4�م��������T	�	�+ׁ��ǯ�57%����Jy��О7�-�o��{�to~�<[@�^�ْ�Ƈ"m���%2�:�+T�?�-ꌷ6��;]�seg'82����,FC�?�Zx�L�U��#��Kw�������R�[�7���N�=�9vV��;�X�;;'}�2]k�*��7�u(�P:�N�l��i��Fj3�ݧ�nh�Rz�5pfj-�H�������$h�J�OB�@FD�;�/� 	Z��@dE�������
�k���E����zjPY��9"_n=VC_�QS+2E���]��3�ԻB����Uzf���پ�|J�%��%��v*V�*J�z�>����(V�3'��'.�����Q�8�c��P�_�I��Z��9��0Dn�[����a;+}&�d,Tn��Ptdq���.K��F����I�+�pJG�M���Te���{ݨ2젠���y���Ga�X́e��8;����ﱰ�|����.�{'�%ܧ�W�CGSOf����d�f�֣���}A���@=�f�Rg�_��f��$o3�n��7,���)	��j�6�1Վ�g��'��i����L��.�HZ,ǡp�3�θt��aH^*s�.W�\y�~�����%H2�@�:Q��b��~u���Lexi�zj�'�<kcZ|�e��˟Ի���O�&��>+�5�u�!���E�6����Du3�C��+gM�����rd<�Z��I�(��'}��wP3�o�i���N�o/[iw��&OO��EM��9l9bX�e	C�І��k�c�n9RU+2e��I~��u{�r.�Uc���f}�u
o��5���O2F�e���E}AU�n�l}��x,��gn�c����g�-�S�P�0�^>��6�"2Y!B�6��G-�y��i�6v�质�0�##��+Uz��q��'�!H}��Hߟ^E�R��V�`M��XL]������c֥�w+��_�$K��K�R\08����L^q��R�w��K��;���1���۔��I�Vϵ�OG�(�+.���Ș�Vʛ㑖x)�N���g��Fv��x�@-ݷ	+әPx�E�|��q��c3���:�����b_�U:/�I�z��lOc\�T}I�͗$ҙ��������5�$�'��,�s�T7��WN�Se�S�=��ZwT�#Q�F�d��t��l��.��x�f�ok��>Tr�UkQiA��e���3�ZF�g��ac�������I'�\�}ݸ��u�Ff���Bkj^h:�e}���*YX
��O��}W�u�Ø��n����c����=��諿�@�[&G#ڊo�p��ضul�i����b�8��˨��k�鉆Z��H2��~�q�⿋�:�>���~�9�Sn.�D��f��<�;�O���jh�^�KȬ�A��LM��#��2bݴM���!U���ӷ
��n��5v�15���]>A�G��;�:e����`)p��K�����mR����A5NlK-[���b\���������"���3C�,�/���H_U�n^]��e�혡Z6ߩ�C��z2�YtD�V��1�ۖ����d:�9�����߷Q����D��H�\˲yQI{�߬�|�s3S�Z�D���V�C^�]_=��3�m^ߵ%�� �)|;��A����G�7Ͼ�+_Pd�ŋP�{\x
6:l���i��fj��<BwY�"�#�.�:+p'E
��$�j*t���nu�չ���,t�ڨ�f�����@̂4�W�
��N]j~-��	�2ؤ���4�(bu�~!�O>�PI}T��:U��Q7�r�V��E5��__q��}T�'��u�aN]��3���APا���6HP��S��cm�R�T�g^�z�O�q��fo�hʴ�z
�Q���}���O�h��ʀ�~�!�����c��t�s#�$���AF��$AfC����=(���ͧ�ȕ��N�=~�n���k�#�:�9@)�.�4s�|��y	f�����Y�oqE��7���z�3EyT��tOR���Z__��R���aQ�)�X�YP�פ5o�/��~�KK\j�[{!�.�zS�|����P��l���Oet�}Cft�[� �KQ�^�D��	ծ`�0�*Hϔ���:�1���X}�Z�F�Y�E較�.a)z݀M:�GN�Q̈-��Z�H�F�!��fM0l*�,N��j{��P��0���U%��
Z�S�\ԙ0�z�u�a����|I�~�JlRGUM�����ښ��l�W	��o���e�f�5d�L�]��Ie�����|qy��,i��Q�Qw��$����J���s��T�xQ��+�7�4�,��M�ԙr�N�r��R�P���%-ȰܙL.d6LC��{ ��}[o�쾲����{�DR�~����!���6��a������Z%�#x&]T�j�xw-]�'�h���1^���V�/j�M�Bw�;5�؜'}QA�o��;�$���|ַ�~�M�����$���>�L��~U����o��V���+�R}�����Z�녘/����v���������n'���;�� ӫ[���Pwר�N�>�d���vA%�}A�h,k
��6�10u�;�s����m��QtF��>`�[�o��<��ב%�	����;��v��/|�J)���,���V:�9�to[��pj��(bUw�B�}~5:�7�^N$3�,��Z��Pņ�M$3f��V�><'�������rz����i��{E�i+��v
Q��ބQ�z������}�k{� �j�A�+m�01�Z�Vf�!�g�b�Ӭ��G�sp���9�!��� �rn�ķچ��J&��ro�[O��jVx���ظ}��i�[{�F�cnM	�:���yX��]1�;#g��T��#h7R@>��e+�$�Z�Y)+9+l�,�Xj��,9�y��s��E�qy�O�Q+k|h5a��-f(��lmI7ѷo�&�W���W�?���3��;)���;~	.2�%m�#"+"Ȉ����

���r3�s#�q$~��O�[B�a�-45՚�4lo)Ϣa�T�D�9�0BD��W�'���,�u��-�-P�>�D�Ә�1��e�Ii���i��f�6CZ.P?���$&__��?���!4�S�2V��Ⱥ�dTXH�D����g3�e�
dɵ��s�7��ܒ���~Ċ-tT����@��o�Ι�q����4��3�|(0`X-�*z���7#��Z�aRjۇ�]�3���@	��/!�vh�>=�K2�M�~�LT�����
��s�zVA�5��SG-Q�i�T����^��l��rk���8��B+��nn~�-�i����|�����0�M���*.�a�o��������l�� ����,!:�=,�;_�R�1�k/���	�L�.��S�^� �	tD��`��ThREBPa�]A�P�pఖ��h������.Z�<}F`x
3��L��}�y��Zw�-��̮IԊ��F���{*hmF�Q����o�rQ�;'"�S��������"jԀK�abҤ�C裍6-@�X��!c�*E��g�T��3�~	*ƨ�ґ�-��v��5���!�.���}�f^��
����2�5('�$j5��a�U_�x�����?*j���^�*��i�2)sFmD�`OQX�A��p+Dy��FE�O�n���8L��������Kz�B>��R��X�˚Fˀj][1
���g��w�����vP��ɩ`�N�)!�5	��ok鄽���P+����5�1�?����u�����;P�W�e-TWPsR��n��[+�גdJ�C>]c���_������o5�F�a�I�S�G�{O�V*܅��NzH++b<����P�؇��|�-�,�l��{u�t��`w���\= ]�̪��.��ר~�g��W���&�F9n;��qw�̱F��F�R.�Sӥ�:+xR���ZUJ�.�k}�yn17�*�����RZ���Q�y����idJ8�ª�D7�hᰞ\���M1�w�����K��ն���5�"S��a���� ���dk0&���!Z������f����x{�@���-�6ۅ�?�v4��1�^�����;�hE�3�Z��|��"=آkǹ��hF���xF��1T�r�Pe��MK]T\=f[�sŔ�~־���Y���{U����S'���<X��J������{��h���=F#?
�����P���<�=7h����Vo�MsF�C��o��`�?gw.^��Ф�֘m#�=2�S��c�̈N�3;��Wx2�_$:}��]f=��]ԌN�=�G�*)1�j�_Cˌ����Nw���>J���Ju�^�f-��\ԧNI�[����`�m�.�8��گ�Z�Ф��qy��d{�s�JT0k�Y1��t�5��d~ {�.�U7�7O����*�IF9�^TM���i����J�wU��R�*4��9�\���6F,�ё���=�FUb~��+�1=w���/='�<́�Ima�DWG�/BMK")�:ԃr�Z����nכ٘cr���IX%d�0w7p�����Sܦ�`]��hc�&;�Z�ӝRR6�с�@5���_�e��D�1;�,C(�&��1�-�P�?�4{�?�3���O�0�7�����/��ÀPo0aNN����	Ƽ'���V@�Ƶ�b�P�$��z��Z��%���Z���xD����fiNM-3A��HTB����\�-I�w��Q]�%�j�hMv=�t�4,g����Χ��B���	|`�����n�����e����>:�Ѩ�QY�����t_/jM���c��a-��V�I$\Rj�p22��o���L���-o:B���$㙣���Ɨ���.�B5��~;�6\����%:�8S� H�����­��el4$���r�	?�H~�C�IBE�*C�0��xC✚��R��J7�t��k��@mac=э��CMD��}�jq9l:w�gůчlp�)SZ/�A{��� աM��-$��,��s-�o�*���� ��b���]B��&���߷�Ky��S�����u"%�����7�{B�T�*�t<^w�/ֳZ��C�~��k��I�R��=��������Z�=�g��1�U���0#��XX���C��`5k5��}�}}�C�X_&AՎgl�kE�U3L[���0�"�z���EUaj���o�&��g%�1��½�����HT��iwj�t�>�#���~'t�'\��h�c���:��Z�h3-7*��[Ք���뷒r|L�j0#�0��fYG(�`��|C{�[�`�O�U6F�������������H�����,|��H+W����й�RW�aT'��1���|2��E��侰��珃Ƙ��lCZۇ��~`kHC�Y:4��8P��+3���#LN���j�ɉ�_S��Y��sh���F��ί�Hk�*x�L|�
wj�*��_�L��rR.���<��VCI�`NjN!.m�!�����(��AGލ�G綄���1,ehSU}�'$/�:�n|�N���|�3�+���tfꃘ߸I$�<���@l�f�sx>ΝL�\Q��ϩF�4�%���^�@r̔�Ns����^���.�0�v��S��A2��%�&;K+�R���!�8j]v���zM�p�m�qހ��R6���B
s�g�ҹ����$f��~��-�J�z�b�ұ�=����{�T�B����m�S��gS%Y�`Q��r�+FL��٩K���;��X�����K�S­s�)�kז�H���xM#|P���t�M��H3mV�����=F�Q�e,9��������h���mF�:��B�$`T�H���X��F�}o�7�h;7��r�А���xT��"�ݥ��Xh���t����g�#uN�=ZCWQ=rG�Y��4�
.�>M?a�����j��J�x�h�.�8g:jN�����Ls;��GC��D:ܦɤ�,[�XIw�� n��N!­Q����	u��2e5G��d��2�i��s���i#=	8����	1?M4�r�y��s4�YrAm����9vq�Xhbҥ�(?H�tRs���Zf�����$9_���l#��knk�cS���M�	!=�!>����]�²9u6�)��$��&��;�����eC*�	s~��B5�IP+h�+��XQ?&9���|���f��	��|��7��c|l�rܵd�7ן�ڈtLe�/��)�G����IHF����^�5hu	j��`,o��%�%�҃���o,�)tI�=$�!�z�����x=��&����x�*�y)S���+��a��f�5<���~5t4_���]O���c�J���dFq\վ�ar�@腬� S�|�Pz���F���`\))�!��Et�F�ܹB9�i�=���O4m^I&�{ՙ�1��VԺ�j��[M{�mK|"������*�(?-����sVd*Iw��p<�iN �������f�����o
h�#��obUkj���V�* c8�G��U��½ҽU��ŝQ6�+[��uxm����jX2�К���z
��wNIGއ4f��2(S8��Z,�E���.9e��z�Y$�F��X��<c�m�53�&J���<��@�E�����sPr��tQ].�L;_r�s�x�����A����N��LQ�) ����<0?�<�c>Z��۔����&�&gvP}ID��o��b��ˑ�(�/ш��^$<x��Yw�[���	�`2��̈��V�4}%�Ð�L�h�)����)0�S"y��Q�R@�s�xdd��E�f8��T�y���H�=��Ǳ�hԇV޼�--��{�QV���5�RFv�"�r�
��L��k��~�SXTy���y��v[��٣�ld���ߟ?��	t�К���ƪ0D�>Ox��^;�S�Z:~���z?ʓ�D2vқg�
����4�n(M-_��ck�t���&2�`�1��6 $�'t�=	L���J9gti�w�,FBq���YD�$�W��q���9�Z�^�./���I�|�XB
���f-��>]������"�ȩLsǪY�3UԳX�dZ�J���'��<��U�2+��B��Θ�c[��FWē����*��Fi��EkiC.?�*������z$OO�d�*j;��⇙3A�b�nG�=��&Ig&��J>g��vf�ql��#������`�^�h�Qe"���A�C
{5�}5IN ������A
��C�BL��I=��Z��3��=3��а?��f��T��l�t�C�HaӑTK�*]��_��f~�n`�u�.�]J��q�K�y��O3��s������=����Ǥ��~���D��Me���x��|�a���J�o�y�cWV�m;�`rF=њ���ah4د���v-�kB�z�d�s�-,�X
����*5��˻Лk�}Qٍ �9j��{��Q�H��t3�8g���ʺ��E���;���TZ*w��6��lͱ�L:ޥ��D���}Z��{�ҧ��b�i �x�T*&��
�I�l��.J�tݣ�����M%�@�?:�74��E5�|h�ƈ�8���M�~�̡�;����@Kt���[.����;�������R�qE�C��e��Nf۳s�����b+mv����ɭ��ƐW��>���3��i��e���ң}��a�4��[Q�f��N����ˌy�6�B�쮏y9���B[���<��Z�i�:��,~t���%��Ϟ��G���\:Q�b��$Bx�A�˺�Sj=8�����RT�$q"��q�}�J(-����t�Ҧ�{�-�4e��"�j3�+������(�j��hT9����B:0�����"����t��8k�Z�6��xiڈ�꺉s�&�M�c��VE L�բM������4ct2%XI����"�]�"J�����E���*HY3�zg��=��[�/}"ma��j�%h�[��tE�>��cx�F�����t�\�"߫wPѵ�7̠{��~���Q������:s cΥ��+�ۛT��n�DA�s���:R�c5�(�F�Zv�Q�;�+G]ʣ�Ix�X%��]3l-^�]�h��p8&��j5�#~��<v�>rZ�Cw���^un�9��������,n?��l�����.!���y��/�G�;߭)��_�}nak����>oYk*u-��sO��o����Gq]�C�W<Mպ
'�՞4:k�����ݙTh=k`��h"^��r��΢����g�/#GN	�z�&?�㒿�^�j=C{Du��v�]����˩�����[��j!VYjn���?��X�8�Xk'�Ƀs�I~\Ca�C)����Q���]���#}K�=}ϋaez�[�7���\�}h0�)�'Z��u�>0tW��0L�̠+�e_���j軿v=K�G0���|[N�=vSE�Mx���{l�H����S��0����|"z�u�yg� D,?����f�-qbծ�M���FX;
�n���GA�h��k�$�%��b'�������X��T��@mT�wm�$v��?:����O�#BY�9py��j�Ǐ��YF�����?e��Ӟ��e��-�-�0�v�%�����O�^u�2
�u�P)���6�ad�V/�]2�Ȩ�J9Pg����K�d}�S	�znΟ/���5�3vjjvM�d9��]ae����uaT��	��dJ.�K��S�������g����6ƉV-�U�e-i�!�� �oJ����-|��;?8(+$eT���2D����PC���{�Dh5�zТ�ș�a�p"S���M��Q�����6��#�?��\��6��)��.����d��{Sѝ_���-��]�L�,ꉩ)��b��6V��)[�yXu9t�E�7�Q�zS�G1A�f0���ӂX35<����z��$*����S��z|i��i�?�r��V�Z^���Ue	@�i'���bh^U{��PX)
z�Ei��>,�}������`���BC�EG�pQ@���\��HR�~=���yҿ/��p�y԰�����@=u�{�,˧Γ=�,����/yyi�j�V Z*YB���#Ui��9?ae�<�K˥4:�;�$�e���Y���[i�[���N(>:Sc��H\V\�Q�K'���Cߑ�s4��W���C��8-W��J.9��5R�aZˡF����J)��Yж'�W����>�P��i�7��U��ö�i���=�A}�NT�t^;�2]�*���B�@��K���*<�ND��4/Du��ݺ?�������e��Z_�E� S�\u�P
�W$����g�<��@�Kmc F��gi�f��*یZ����tLIg6��n��x)��pzrn4���@�7V�lp�1;p����sǎ|ec�����V��<Ɏ��z�~	K��)� o��e"�n ��BT�k����z⢂�������a�)(�&�(Mx�7�j� ���tQ�-/Nw����a%A���S�.��\B4ǰ�(�1�(����jQ�PC�	j���7�1d�u-�H5��	��IӰ�T�O͔�$��!�\!V3P?��@��`���"��f%Y�d�w���P-l?��L@�JwB���r>�3�@ݫQg��?��&��Xą���R�nF���ڒ\�yW{���vm�l��܂�<,e�Y�Gu:H?�@�҅�*i����rf�����a��no%���JP]֥V��*��k�D�ci� ktI�e�q���8�Ԇ�׸��:��D���j�j��)�Qa�o��[H:�%������ܯ��'�Q�5@����;����r\	�X�� ���Q�jV�����������޳{��-��	��9�f�@��]O���O�aLx�L�����?�ma�E
�����������Ub����CԻ^�Ye`E~H!1?�)�`<W�P0gSȟ>�돚ޠ��<�S�6E�W��*���j#��7�u�j�Qstm�3���O���z[��7����-5��B%
튈1���1:錴�t�^0��U5�
���kЎ��ٖxu�z���/ZPE�	j�maZ南pPѽ�zQ�a]QԴ���a�)(�ju��}��n@�[�*���vPwx_|���(�]�sC�]Qѷ=���(h��(�L߾s�W4j�W����"G5�	t��7<��9��e7��q�+:0唣�l <����rf~u��e�Ԏ`�9�er�o�P�-M�p�����P�Q�u�ܡ2ې�N�{������J��Z�sA]�&�.$[T� �ɨ���vQ.N��M�^�6GUV���^�w��GO1r(������u�H��@��1h򾴌���L�Cujƨ����Bs��<f�
F�BP�����(IA��sa�ZE��n���'.
*��CW�g�� 52[P��1^'��o$n�;^�E?��ʍ��Jh+Q���@p��������d�pE-�x��p�U�1�5N|(���`s(z� j����.{���(RPk��oAe[�.PY{ˇ/�k���P#1�q����'��"�U��z��q�#�+_�P��#�/�ʑu��-�үt<���w�D�f}�t~<���0
ʍg�v��@�n�f��T�g�?Z9����, ��
]X�ï��'��`W-�1��.M�)Ţ��;��X��@�[Q�\�:��U��^[Q_�̿�1ש�ϕuс�*R3~�3u����[Y��2�/D=�]�c)JP�>�h~�"�{�����1@m�b�qz���f��ۈn��CS�lQ�u�mڔ�Q��G�^�5dp䆁�)Og��ؠ�XO�LP���̯�Q�NPmtDQw���{�HiVG%�� ��5���g���!+}�g�\׶���&��JuEЋ����*���$Ef)�S�w���S����2��ރ,����i��i�{�r�UB��)g�L�
eM�"��{���9ۻf�/l:����t/�

��?�,��?�JG�Ϊ�m$e(%	%9$���6=.қ!���]���[J(ǹ+ZT������!&��RTRV;��r�mP����q��A�}��D鮴�i�{Uu��b���-�S�n��s��$��3��n�!h��;]ݟ�h5��Gr��lkyDU�B���5��9j����
�v�t
ԃn��������ִ3��ߍn��Q�)�6��˥1�7L�c��E�3�m�f�X����Nj��~�{N�1s���+�=���s�@fM\N����5����n#��֦.����fjj�q�[��^���~A�ʢ�<V@ &�5e���&���\�?}"릘�I�F���}ȭmd�΢8���>~�1�2؅5��mW<2�]vs-��A���¡�%(��'�&Һ���}	4�$�Ǝvr�����oή>�{�'t�#!~�����K߉z$�����@7CJ�	�h%�0o��a&UYB\s~�#�t���sg\�#�2p~!�M�o�_�������R��K���m���pJ�x+9���;�hm0�0�!V{�?%�Q�o����>YQ�7�ЅZ�[|�}f+z2#,�-yO>�Uӑ?��]�MG�$h�E:�2�L@1N�W�fZ�}X�7�#�7U��l�7�^]F"�4�F���
�[G���S��_t�8���`@�_�?H	�TI�F:���}r	�\a��Y��r�_��(4-�������H��1G9U��%NQ>A�=O��ˇ#7��;�J���R�K�y:�[�"�ʔw�M�������Th�T�O����zg?)Gy��fΔ-�	km��Wvs���z����� pd��<��������$�����X���!a�xN0v��%�0��P�kfU���N|�=��}�	�K�xE#�6!��J'E�b?v���5%�E��!izy��'$�GZ�]�5����lw(�[�r\�P��^�x&�d�K��`:�6*9adCI�=��L���uraT8�7	�|P�H&��:���Vni�ףc��o�SE�A�"ƀ�?�v.G�Of�E$��d�����g�k�����;Cf�~��!	�C�85/�+�E�@<bh�������!�g�>�K�j&�_�+{0��[��y�li�F#�Nd���1�Fo��u�	�Ɋ���XK��32͇�N�@�Z�2�����2^!��Ii����[����m�}���o��,5{�nIϫ")��.2玠:��FN����6�2�AFEې�z�5�����VTh~���i��Y��SIrp�~N�a1���0�����5F�Lxu\Z4�#=��;��Ґ����rz��_N�; t<l���~�#:�I5\���HQQO�L_�_2CU�}�)y��<pb������n����z�[��l4�#IB q�X�O*}����p�1��}X$��w�Ԅw��4'��+E���C�5 �2Q���7}�l�k�����_����4�n}�}�~��ᥓ�&P2We�@ײ0ɝ�p��(�ŜEQ'^B�,���KWk��E��B�T�,T ������@q���A�n�{�@t��w�8����'��7%�vC�b5Ӱ�̌�29E����u>Gz��ƴ{i?�3[b�2�WT�?D��C['2,� �&���/�N-I�ґ�� ��#���P��c��w����}����x�4u��J}N��'�Ok�S�aQ�� �����j�[6��2U�ҟ����|��%{��EgH&�=�dWH�:�?��?E�ad2[&Ç�[��H�	Gd�BVH�L����@vyؙ|����?�'�@Y$O�Bz�w��b�)�+a+����u,�v�Zvθ}F�˗O����kB�\J��~��Gl>jtq��7X�cʇ�Dh�DD����>�$����8�Bw�r�[�dO�[I;U�F�%�/�RO�_Y|��"�z���y:��#^�g�Rk�	���m�� ���Bt/������W�����)�9@�xV^�{փL��������O����E�I�E�?7����R�r��<Ykx�Oi�<��J�hT����oS�M���g=s���k����R�0c6�Đ�*���eA�v���5!+a?�G�	z<�:j��B~�x�4�%r!����_�d��J��"�>�E�bÍ�F:�~�,�3���zg�>��|�Ռa!S�N�K/W"d^���(����e�!�R���c��ju�r�|+V��~�{������[-bԶ�|��>��Q'�$�TP�&M��-*[J�m������AZ9�Q�#�Qϰ�{6����O���W�Q���^C_�'�����Z~�:?�5wP�W�՞�q/��=��O����̘��tQ���M{�N.'`5|ksYԧ��P�g(�'mB��Sg~��\��Ϻ�~��c7��6�
Q�����*+«6)߃5xN����.����-AS�V�,����
0���J�趴�!���w1��_A*�$H�?n��f�C�U���4F�8::Qm承(K�1�\ڡ����xi]7�<�.�҇ٹ3���F�}l3A3��xwk�pg���-.1Ucl����vÞ�k��&R�hp�=�U�uՃ�;��B	�v�XP���Fr�}⠾�(uχ7�N�M���,z_�تe�~-�`M��gzI�7���Ff���	�i��Ї��j����T�>����/�JЉ���&-�$�z�C���.���Ga]s�(��F��zBt�H�{@c2L�qc��]��|.񊩃��gm2�,H,G��w���6�sW���B-� � ��3���숞دے.�Ct5�^�n�mM��L)�l�3�GD�s�kN~AE3��8��C��t-��킆�\��a��{���:��;(/�ۨ��e�y(���݊ѳi_�Z���p�O�8�>��(6]�R[\oG�	�h�-}Iȣz�u�S�MK�2��|@��@f����[���e0RK�+���h�3P��y������Pk�4p�&BS��9"=��ޔ\^Iӎҗ��H���P`�������Q�<�[	��zy�������`MG� �{:m)�UrN��ۊ(��9�~�����ڄ�3h�O�9��	�|��ȉ5��X�̡��ջ#�[QR9\/P�>���V���5�`J9�<A���Se<"��>�+cn�^!-�kYq9����zxcrvn���]��6K|28��hNa�h�\X���r�2�i�(@�Wpǎ��v��Q����k�j�"�h�T.���M?g����}W�z��p
کfM�wm��L�ds��Zqt��|0�	lX}��&^��8Q0b;2"�sF���t�x��t�E	���&�+�,�C*G�$M��JN|��������I�r�s���#�ל)�������K�F�L�UI���w��O�H��c��;���4��f:�r"�I;f�:&oD�v�	��U�ለ�۹_��OO�mb+��*?�9s�[����G���A��s^����;k*	�1'JR��j�?�������E�� �oY�%��+��C�Z=@�J2O��-9�Y|��~�z��@c=ǩ��̹��\Cx-.���� I���)B�r�5+)�Q+RlzxA�eZV7��d+r"d�C4b��`�se� �O�8g���_�xC\T�z�!b���X���d�(Z����j���������$5X�7e�����H���Y�GA�P��5w
A��x[�_dԏ��`�
��#qb>j��9���i���	�2-j}����T�뼈� L�5y%|��=�]X�L}��\;wP�w�2���Ft8(ץ�2^ߜ���L�����T�z���M8�d'�rա3��:��r�ӨS���Ҁ7Mw��>I5����Rc;>���ղ����Т��,j�[�b
k��0�8���<�n-���� ��j³,�Ѣ��.KBDW+x%��"kuQ��N�E��hA6�z�z��zou���͜{�[�}����sϙ�fN�T�W�	���֬"$i�y���;d�e�
,W�~3	��s�	݃��+��BWMcH7�5����E��q�lr�q�)B7[g�=�������P5�OG��z\T��5Y��苸��s��g���6�;s#�p�}΁k�ò2JB+��E� �q�.�Q��$<"ڗ�K̸�9|�\�W�z��6�~�EI
��OVQ�Iq��_[�~�G�)Ew\�>��H[�}�S�¨�a��Pϝ[?f.:��gi������y�}�}|Y� �s|�BW���b�6)>��YnhΫ-�rx+�c̅�))�_�D���m�DW��S���F-9сB����m����ȕz��?	���kĹ��Rl)����ۥ�%��q?����\��o�u��h�)��y5c�`�6��O.R�]�"����l��m���.�����7��O��g�|�|;���B��W���F�#��̪n�/?�̢3�T����n9�FD�O��ט�%���N砗Q��ߘ��h�N  ��d�8��]��k���k=٥��rМ�s�0�-\�����Ŵ�t��	@4D ��KQh�Lƀk���s����V���4_�Y���K��ix!�󊔺�q�Rw=F�0���L�[I�*�O�k������������L��C�\�s�G��M�p][��X��a�hB�1�;4}��[pvʢ���1ki�!̜)$�GqS8�#�pm��ɜZܩ��>�=+w��L�\�#At�Щ^��Up����9�d�6�*�^��t�4ģj�$����r-7/�g;��b��'�'���U<q�;������f��9λH�aˀNv�8���P��\�O���@�Q��@��L��w���Hu��+&=*�c�3��lBh�p"��*"쥿�?�� 3���'-�F�ʳ�dO��	.�e��f`4�*������Tل�x�����13'd��W3��c��EW���+o�a��x{݇����0�e��Nrx�:Ǔ0d�0F�r𰌜���au�MY��j�v�LӘn����G�J�l�<J���\1���(z��
��Ħ� ��3�E�vJ�I�ru�������m��6���k���'�>��s�nU`�1�g�y�����C����i�`r�yHodj�b<ۯ2T!M@6\c����P�F���>��_�'s��+�N=�t�����h/kjdWԘ�9@�;��t�F�i't�گL6t^{�yW+zǜcz�lA@�W�#b�Ѣ�d��SP>�4�D_���@`�؎,I[ ])�`J�X�����Q̦(�ű<ސ��'j�G�h��qn`?����E��8�&�Ak�3J��eCtƓ}a�[�~1[,�����c����k\��{H@�5��k���Nʤ��X*�� ɞ���%���vݲ���]����$$�Hw�]A�v8yl��ޘ��f�����0iHܮv�O:@OZ)T-��׸��vř����չ�F�*�&ds�M�{\Ia����5��a�Et?������w"��x����3e1)��M�a����0�%pbY��G�~`�[Ύ���r�C1�+W$�6�T�q���G[Fܽ]?��$��=�g{�2��::��ڡحXFd��+f��'�"[��*��w�#�Vn�n��<���W�"h��8�Oי�;r������+6�>��0�Va�ʝ�����-NyAku���4�xj��O��ZM��N^�m�Z�.�Q�H�r:#<@�2\s�]���0��q�p�AJ��i f��?��5�
�s�-���3(v-�z��bg@��O�fQ�b�s�Rvp�t������1�-o�p�������3^���W��h��4�5�+tE�o%���ӟ8v
R��h�b	��c�X���؝	%�=΀D1�wU ��-��<Y�q5�o�aQ�"��`:��MMr��3j̡��Y�p5UFd*\w�Ӣ��,�V{ڥ�u�@]���𜦺\��25���<Uy�r-n����h����ZM7��Mn$�2�E2뤧�И����u�c�e��>����A�k�i$��@����pD��LY�����cw,�u�TrQ��H��Ԗ���׿��s��QLL�)<�Q�4,�g.�2p|1�^�UU�������eޮ�<�9k�1�{�v̈����������f<|8E^ڙ�nQZ���w�Bo�ܩ�>X�쯳 ��hy�H�˪�jF�|�P0�r�a�;���l�*[�0��_�O(�A��pA����` �."��ҟjo;F�^�M7]�n�1��os�h��"�v�S�O�r�d��d������r���I��Z	n\��U\��t��L���e��{��Y�$r�3��Sh%/4��7��[$ �o�qU�fӗ�Z��� ��c��{q��r���	dVy����Jv�PF]<�oW�s�T��ዻ�!I��ٍ����;��Sżډbu"�F{؋��g��yu�C�L�aې�hq��.^y9MB�n4O�u���8h�\���iFD�]��]i�W��TĦ�R���tM��EP8�� =��?~�����O	�b��U��$q3r1�.xc�@���T��U<L�Ul.rV����.�Z�Y�v	2kÄ����T6�{�U�Wͬ���1H�r�H�`�*!��Z�,�׬�?c����m�|b��}5��L�1v��Q�J�f�0����݌F���������jB�%�{�"X%�z�}��=�;^L��8�D��Em'#�,�2|���!���*(��L���h}{'H7KXI �%�A&��k�ѣt���l�K�q�=�W�$�+\�;��	D©���hꬼ���X2��J>m**q,��!C6�~U6q��(D���~��4(�.m�������´Ԧ�?��Ό�ע(�!�2�F� �H�1!���'3M̰c�?d�A�vS3������4�Ľ8P0�������(��m����u:�x�\�J�n#�VERw�u�
�;�6C*NP��*�Y�^�O���(�4�A�O���f���_�C
��FY��@w���_�T�5��Cϱ�*WX`�0���oV�Of��oJ;��9S-)�z+�����,qi]�������y�Tͦ;�b�\��JWۑ�K�^q����S6�B�92�ފ�b(⏿%bb����]W��1��!�3'����V�"�"r����о��+è�H�j��� �@6�*�uD�
�5ۮ�\X��]�o3������[0g�T�,W��p��D��AW���e��}F�}��W߉
���\"�3�~�՜�eܗ��:�K��lBt�	��!_�5�l-�ik>5��PI����L�	�s��]']�i'��4�i�6����e��D��w�#�a�n�@3u~5��rf��
���/�|M��������S����f*�1m��(�>]�yB3�� ��3M�i�1�?���uu��C|�A7��p����GUi��|��k�k�*N9*\Wd`È�|�2��f��no��_�k�����`Ӟ��CT�q��v������/|m�e��|MW�+�nc�뤅�d�e?���`J׾�>�6mݣ�w��{�&l��Z<h^�� LNo��\���5��M�����~�O51��!�ڪ`l1�t�������~\�e�'b���K���|��Np��n�ō���DEg29܈|�K�^��q-ɚ\ �!}���Z
�i;� ��A�����m��.��SO�C�o����ș��|�C�M��)�{���A�_��O��C��c��X�����%i����e���1^��{�HH]�a��c0����#>h��i25���n���Tlo�+������4�Z��6�r�4�G;����׺��9Sd�tBᮃg��%����]ź\�Aȅ�w��	��ө6~i����kR����]h�Sb!��������>[���C���@|�o��H���b9�3'&�ٻ�%z�ߑw+%��#���]w.c�+ئ0o����P��LیBS����m���G�D����;d���UW��&Y�K+;^\�6����w{)�jB��ڔ��z�3!��d�|�գ��|O�n(+p��)��_��(�zx��d�Sp^g�8�Y�Um���!x(M�{�����x���P!�盭im51O�٨�N�.m���r(�WWA�O��	��Y��]���0��s��p.�#]���k\9��J7�!�(Lt�� �&�m�/K)�4*��'ь?��{=%�h��q��v��4C!�����G��"9W7tv��;4+�H=�`�kf,�t]�4&����n�jI``��n��Mo�j���}�w	*$>�n���~Ӻ/��B0�k˶x�Zs�������x��G>��\��$ۜ�U̶[K5�S��U��l��Ԕe���:<��6ցTz��r�;�Bg�IC6� ��fu����l;Rjyx�F�@����eCb�)��}��:����a�ȫ�$���ϡ�[����Z�8nW�"����_9Yڜ��s��%x��u=b�=T�z�w�Z>������z�*�Nk�����K,����<�!�\��1��U�xC�̕�/^h�q�&_HO�����ui+<�H��l�ܟc%��<�&��$-q��g�d�B?�F4�v�8�ޭu��q�O���U�0�����G�r����..ߵ�<S$4������/��]�Lӆi��5�}�g&	�`�I3Ă�mG�~yu�v^���,u�^B,���[ZXl1�m�%D1���*xf[��	�O��V�!���q��b�SL,R]�nA��ff�%�m��R��0z
\F-?bf��;r�Y{����g��J"]�끳T�&���p:{����Re��h��{f�0�ZG z�u�-V��[����R���[iQ�ЪX�L�3I�"��NTۛ��LI��R����X�Ŗ��M�-�_k�P�=	nI����j�K�"	d!6Y�2|���6�[��!��u�9=���,3���#��=�B	��%U����q�� B����m�v�w4�{��l�7�d6�����M���% �ȥA򨵾�ٙ$��ԇTO�.I��;�j����t�.\
J3���k����k�ρ�KP8�b*l/��]���T��C0���
.�D]��V��$���V�������\w:m��aw��uc��'WW@밯E�C�i��a(�V�~Kc
p��w�#3�{wdJq`ɀ#��nk��'NEZ&ڶ������oP�1^E� )k�mu.SkH���Fb�`�Au�P��q��*Ԥ���&���ބuĆ�Ȅd�x�ծ��v���^����tˋf.���©Sʎјo9E�2�*6 ;�D����;�n�hJ3a9��IL�wHx[{��9Tf����-�<�s�]���>è�p����W���!�Cͅ�h,�R���)HK}�W�\ǽ���c0KE�����Mܪ<��4�K�<�J����䱛��m|�����O�n¢#!�k���>*�jyS��iK��u��ޟJZ��[�U`-�ҎM琍����V�T�"��4pe�I"���#�^5��[��Wc���v��E��4dC6E��Z�;�����6z�^�S����$���;�IT�ٜ<*G	N>�� M�����ʖ����l���0�t�����y��Q�^B~h��8�9�ؕ�����*ҵ]O3D���|��	���%i���ֶz�۔��Yt֍f,�m�-݆k�G��K��x�)�)�$�x/������G�u��U"����ժ��jRM��<$	=��a6<����QGP@GA����ǡ��W����0ݗtJ8�MA6�ۗ����w��z���(��I�bߠ� ���"��ò��w�x;e>�=�o�z�=L ��V�.lӣ˄ٕ���&m��o�<3�4tʕ{fcC3'�q�B��p� ���9S�sT�����_Q�ـYpd�G]��+Mϑ"���	��`�
bT	P�΋�l{�D2��K$3���^k����P���T�h&��5�-W�\�N�o��WK4x���]ψm&^K�ÝO��8ml��HG�������j��j��3u�v����OJ��g�F���8t�X{���������I}����#T _�3i;��ֱ��*ɗ�D�e��u�"��#C�T#'���ms䏊~�7�IW�j�Sʄ���7s��K�ʶ���?��b�Ni�6�m�A��Ry<�5��_��N�o���Pߛ�0﵏�K�t'-	�g���i;2ŶU�|�}�6]�מ�\w9��Ac�ݙ.�C�ź�]����ѭ�jߓ�^�ZpD1����Lp���'GY���ν:U�G�Átם̹�2}�j���C�CE�ڇ�?�^W��T~ϐ������_pb�uf��\� Wr$�����xRT���|��TO�i�c�{�?8n��!�t���Ӷ��L�\�"?�n�[�T�,�Ų��$��M��S�$5:��;�������ׁq�ϩh#���3f���6i���:,��l�������}K��)��I�����ətzS�]��P�
�J��]-��-��\�4K{�S�f�1-R����v"�;F����XG���F06��+�g�oQ��1�74������<n�-�_Wi���i�\�=p������[f,����{vק����C��M߾���u�$�7�}H?�5����Va����[X����؞� S��R���ƾ�t�5����&��Ld?�LOb��zD� *�5\ջ�8����ST�����M$��4�4����[ &�כ�{[��fgO(z��X�d�� n%e�IY]H������&����6�V�&W�+U�uE�����v��_|�oLq� ;S�ZC�y��#��A��nb?>�����c	@L�'b'^>���!�j�,�cO�2�l��3�ou��Ku�9�zٴ��z��o�ly�Gc����c笙Oҳ�("����T@c�ν֩�x�z�4;S�7��](��#hrs��\KQ����N�+��"�������L��kdR��]h�NȨL¡�"I��/��+�3k����ҿ5��m��y��5����1���pm��F���t�ꬂ\���(�(�<D"��0�oځ3C�瑸m �)x�:ߍ��b��!��h��O���G˛P���������T�0���`�~�[�RT|ҍ� n�<�xX�'�M3���'}W�0�����%�E0c�U�Vs�&�"_�L��B��� 4�p=;maBA#̢�:38?�(��+X1�	��7צ7+eKtsگ��g�r�K��e�6�(j�/^�U�]�$SN#@/�crg:uc�*�Y��lB���t�ډ��1SD6�R�=��D�K����c6���M!�`o��C�n%�<����b�S^|�0����ГT�oe�Q r�%�~��Gٿ����K_[/ �����L'˪5��M�=5�}������{����������R�Ǜ-�'fU�i��P<B�v5�QSX����H�'�$��ݚ���:�(��
JD���ą.�TV�	���H�t�B�[�����!���[6���H��Ӯ2����gu}<*��ɠ��D3�$��L��ץf�^�2��_����ٱ�c��	-�k�f��.A)�b+8B��T��㕮y��h��A���>�9����I����.6^P������u1f�RV>Cqa�6��f�K�dkfE���Qr���,�l��N�{��!H�E��H����D#�Mym�.��BH��Gf������H	���#U�}lNgSZ}A�G�cl,��"���6T}��q]k��`�wW�'ƤMe�77�k�	�SY #ݥr��[ny�jꔄ��?��5�i�E ���n+��q|�4�����=�����S�>c*���n=�yJC�5�y�!�;�Y�k��!����F�q���r_U\N>�(���wU��1�P�>~iwi��� (hF��i
�ț���F_�'���,�v���K���6\��G���
؄�aU!��UBM ��=ܛ�O�ځ��m�v��ͷn�C?n�Q���ť�<+����m�.ө*�~�F�syaW�r#@����)4��T_:�����K�YU�I��wMӒT�~�<d��p\�F�u���h�4�j�xN�z��,�N5Rߠ��S�ً�=%xZY_s$Ť��X���%����.��*�.E�sa��B�/����7]�́5�k�%_7���=��2AԪ�9�C��fdO�$;b
�K���f�7�BG�e�0"�����|k�@p}+\�e��k��'S����ϵeJ�����(�ߴ��.�N��A���WȔl��oΎ�w��J�B)p���~3�Ej-k�ca�b`�$p�{>`�T���Fy��s��D����G�	��#wv���rH(s�=	\�f�4��?���r���q�������)����JT�$�\;^��;\Yr�|W��j�a�N�}��g����fJi�l6uߞE�'s���g��#�?�U۔6(1=����
.���qÏ1^+҂D݉*��֌ǔ��Ր��0�����9�}&K\lJ�O��b+�� �e���Ӻ�^!w+n�I'���గ�{�A�2����F�k��a@�>n�V6a+��&b���@�J�>��)�_��s̿�����"��-��H�^/𽀙t�#�f� 1!.��A�h�3�/G�\s����nb���Y"�v␚��e╹�l���lN���p��\C2�<7�7�[�ANSa�
m!�F���K����`�:s-_��F��f`s�k׭ML:��5�>spEk霩�.����,5����5�R�h�؇�k�K��"��1r�h�M��g�ΉK�vu�<����J�YJ��ɘx��V���F�F��IF���@ MF�)�g���1L(�1�-�~P��,���ʂ?�IU��T�A���f �P�w���+����4�n��
���I����|ܶ�M@P�hL��״��U�I�D��&kw�S��dj�P"X C�$h�9��������v4�|���G�� UZv��b8A9�� �Ǌ�Q1�o���dШ���Dp	v�b�#^Z߸M[�� �ڙ���d1x)��Œ?Te�ud͑�����LW���ʍ+�
3�u\�~a݊�|?�����.Ae��0ց(��L3��T�@��1T�a+.��k�Jb��)�_'eN�>*F�zV��?/��<|����a�[+8?��f������8�=��\��M�l��'��? C!�u��B4�t����i>ԃ�opq���竽�p�sS4ݷ��P]5g��%㺇��%[BT��-�*�$��Z�1�/����T�7׹M��<�H}�p--w�q�z��-tA�u�X�m�1��t;�+*d��Y=p�I?�kM�=�izhY W��.^����k;ākG�wգ|�ޡR�>��\g�D�y:�5���d�5�5w�����s�r:u�hw��M�z&�1ņ_p�����<(SGс@ݟJNF�4*�|�|��b�%K� �K�G�qˑA9:��A2'��'���"�bE{��b}pZid����Ζ�wuA�̥e�x"̬.�~�)]�N��m���D(*+�m�K�J\ߍ�'�k^�V��������cq(pZ>p�cJ����[�~4�?�H��o&o!8��@O_pO�C\!a���Q�G����`x��b�',� SJ�xKJ�5-̶τėTi
c<Yy�����Q~,A��3�Q��qg����Vj*F�U㑀�H�W�/��r��ߠ2+��_R�i�Z�\��W��SN�J�!�s�p-I�%5/�7M�~��p�.S�z���U��p=��K�q�ƅ�"\���+WPks�ePb�pmI�E0ރh9�~���r\w�O���o��ش0?߿L1�+�N��u �h��(r'���a<qя�qe{�΢���u�4�*rT�~�p��O����}͏;�qe̎���X�sW�9���g��ڽv�U�"��Eߊ�q����3	wTB
�s�T�z�,���ct`����u�\�GD�Y�+9��:z��lAp8\��{�n!_��v�Hy�g{p%��E��(M�u0a6�ܫ9܀��p�v�u�l��6p�T�W�9(C_�z.e3�k��b�Z|�/���U��	�����g=�̤�[q������-o'��iu��$��*��r��`�uO��t5�k{�J�sџǙ�s��
�O�~`�N�v����1�3�S�O>��.wz*+\�j�p� �%4�v*�w"�ky������o�@�̖����ץH���D|�r/�7>��ΰ��%tT����\l���5�ׄ�(���{:zS_���9^o�Kg�r��+#��H�m�&��r��!�������w�q��s�U&p�AS]���:�{�Ї�B�`���[��5)pW������W���ρy����g`�x�'O!�-x���M���Q���d1-��-�������^(W`�!�H�J+��n/p,\��R}M�^��t�\�*�H���=�[�g�7��r��0Sa,:8�ϕia֯7�vF��1;#� ���2��=�q�߉:�x}c/�-\��u�p핁�k\��?��OdO8��n#3�	[֎��>��רΖ�)���}��	:<i��D����4p}�9=z�����+W���H��D�)�'��:���EG�Ђ��
~����6`�B��&���&�|��S��Ukk)z)(��f�.���[�*��m�ig�es�9�r#�+jN:�K��+�3�m���,�Ķ��/���͏&n�>�͆/�a,z��S\޵s��M��&�z�
7|��?%5F'=�G�/Z %�'��1�������C�����vR��M�@zfe�9Y6$Bœ����$h�S#����$u�Â~��pvt��@:�dΠ:�nL��%�%K�8�hD^��ߜ�0�lqv�s��%��Λ�U��97c��Ot����{�e�;K"n����R��\��,�	�P{�0ԝ�*���#��$,��f�YM���J� X6����Gĥ+��a��3�B�s��PrV.��߬+ި�95aQI��z �?�L�Ъ��J4-��BO�"�$��cJ⽓���S��RI�Q�� �JgwҘJd�'`%�<��RT�?����7�����Y� �ur�<���Y��=-���]�����ɭ<.ZB[2 ��1�"F��c�Ь��v^Ɩwر(U������)1ߓ)�q�T3 �JAJPlW9����������8֞�\��T��YFŞ;��$����=ԄTX���p��_6B��Fȑ9���d�'h���<�� �D�|󣹣��*��+\�xo�����I ��g�5�~ӽ|���6�)��Wչ���z�^Wz/l�����M"0���U`.��3�A�m���Ǣ�K��i	N���8W�S!�'}Nh������4����d�c���U�O��9��[<9�e/
����g�S���_(?���p����S��v�Ъ38D�ڥx�g|��?"*:���Y��»l�����l�q�k~�6��;?�n��G���ح��!B�f�翔?סku�h`�l������C���r�L�}���5;�i{)z��ug�ˉ��gZ2�ơ4���ێ�)(IB�/�k���=',�Y��\V��F�P��L�	��*����~��c�-+�ҸWmA|܊�C�0��\�4�p�7U5�)�<Z챌W���R��b�U(�nj�:�T��}�5���K���=��c��R�B�1D����v_	L�/&X�!+E�N��xԼ�w�_�U�tE_h2��UӁg�����\R������8�RI"�A��(
�鱕[���:w_�Z½�NJ3z��	�3�\J�Seh�h�e�c5��765�m0/H�[�*����Frζ#l�]iB<Ppl����6m������K�n.K�Z#Э��S�ue
��}�@Vb�2�glYv��$����!���G�(����yH[�/y����`�k[Ro��TU
�s�:hd:a-��>kM�p�����`��J�)a��KoS��^s
�p�{�
Ƹ�<RM��(ڃrO�ȏ�\*C� %���{�6'��1�A)+�e��9,&��MK�a\��oD�W�0/GŶ|"��̇�PO�;�Zud{���-�[�`�;+>U{���1�"�7�8�����b��h$ߘ�c`���ӈ�t�?Vڹg
=�N޽���Uس���AUv���X�ȕ|#�K�q,hG�}���iV� ��ل
���>������KD�y
�9�9��d����R������#�����ѷf}����-sG���	=u�-��I�F58z��.�9��il(l��ԡ��W;��7�{$�U�xZ�`4��
��f&�d(���X��Ll�+���Y�aQ�^�0��n���a��-�O׵2^���*Fg���e�N_�j�к,�fZ��k����g*�\,����XƘ�*��d��/���3/UC�T���+2�Ȓ瑵Z0=K :-mY�g��3�A�O��G	Ȥ+d08.�����re�����ZO2�,ӛ-j*Rꅬ�Mfs�d^_~'M�+#'|h0��g;��Z�;�4QƑ�b�7M/�3��Rro]��Ԍ�c�V���B�_T"�u��o��1�j�\(��6�p�.T���c�E�V�\�����fh,߂`fF�V�SUm�Q!r �T�{LK����f����PA2ԍ%ٖ�ao�G����-J�t6;���ThbhR`>P.m͵եٟ�����힑+�LO�ꦿ�x�&�/�>���ic� �sԠ��ãC�.t	���_cW�aBc�]�B�u����\����|o�G����O	\��cW���D���c.Ƞ(��ʈ�n�� 
2/צV3,�9�ά���vىB��hbz��>�= ��ξNA�h�{�9Q%�Z�R�3�n��A�r��n�̅dWd[�_�_����ҵ~�[�O���J5`��Q���)�U\JQ��t�� ;��lp����x��3r �~�=b��>�ds��"��E��f7]���o�������m/|�"�r��|�g̵������\c��]#�S�Z����un��3��r��0(�r�v~?-���F�u=̵w���� כ�.�6��c-��i���r�W��,_?�p}wy�4�[8B�~����dPL�1���o���}-�D,���-���A��׳tz���}�b��:�d�a����cy�����J�S��R�^�	]�,-� ��H'�.q��p*�F��쫓o�g�7�GdM����z$�yA�U�B�2�(��'�a�_�s�84J�|�)���޳{d0[Ch�Z��J��.G��)�1�U�IN��tL��_�Ό�']�[!��:#��vdCt�B�d�k�5'�"������l�޼B��������v��r#CI���p����?�2uAo�5B�!-1���h�:� �`����Q�
=@&�!c}��޴�f'������sU�r�K=�ה�'v��h\P�DLӷ�!W&�����wf�[����� 9٨M�/����[�4������U�K]���4"�����h�A5�9���2�*t	��
�6~����C�ɍG����_�/nNu؛��a�h����_�"�-P�w~/O�.phѕIc��pe?�x�~Ծݬ�^C�ձF�[1�33�JĽ<2�A����O{s���M��s=�c�̀�K�I�+K���Ώ���rq�I5�G��˗�������-��/���T�=H�9�J|�28�-'e}�4$��Ŏsȟ��[��![�B�����Ú����O�� ��"��k��V��b���4`����3d�w�D��UjY������?��JLMF&#�峌��	�^���ҞLP�e.!�d����4�+D�)JM������(�}��c��#��^�o�SLȏ��#����Y�_Mu��&�o��6�h�1֑X(�����,�Y�ZuI�@�J�Ke�6� `yo��j��+,��Իn4��	��eYBX
"�����oQ6(����A]j{!�O�p e��ѣH�����̶�1���:����#��:䠅����ҭ'E����R!��HM�s}V/nE@�(!$s��oY�ӹҘr7��_N��ɨE�O���[-4uis�� �����B!dcXGÓ���&q��'k��	M�$5֝ϱ>G�Ŷ�sabk�z,#-��w���Ϝ��O_��q����k��gg�Y��8ҩ+`�Q��G�����sбe����P|D�[Xh��wк�����
�#t#� �C�7���rNȲ��ɺ�!d+Y�מ�1WkX�5ɫ�]���������8��+� #I� d}&%rd���h������[4����
!@*�������� Zޓ,iA7຋�[0��ɹf"��ϟ�b+{��3��P����H��A���*WteQ���a�
�lL%�Io�֕����&����ч�'X�6�� Bo��k"Vb�Q�PcKLR�%Ʊ���@�4M"i<�I�H�|�pVC�$�%8�Z�\pS����+{�h�����h���A�e(�����Ч�R����pS�%�6��U:����?����HlμT��u�H�ŹH�r�;�N�'8g y��J˛Ù�VKD<�!N�8�؛����!����;�~0�w�<�s���߿!�S�ް�0�Q6��8��g�B+��+��Q��	�8i�3����mO��NeJ��o+���mG��L������~K�`�r�vQ�{y�KR�n1`)�ě�'+��&L�$�B7;,s41.׬urJ:b~�k s��:o�ӂ����h	���s
���ᶄ��y3$��
���J���|Օ���-�LA�B�ēQ���Dli���e���
Q�|B�v��P � _^�R�3M�uk�ѕsT��[]@��0yX�4�#�Y�����Ϝ���o�����ٲ+=�D<�es��f�&\�<��N�0��������P�e��R,܉���8RX�^*f9��Fgp'�x���A�׸󩜬CYr�p}_��/b���2�s%���9M��X~B�pnFe���W�^����[�o�X��Zg�v��ІZ�j��q���)_G҂h�cT���F�h�N���b�d%�!f�ɵ��0V�C����"?����ǩ��@s�k(<RsA���qGi. Ww�j�ę��Hݻ}H��+�L�5v>�=��{���d� tJ�O�<Ѧ����⪬�N�12���_�!_����.���!�%t��9	R�f��6ev�Y9��� >%Em�K�ٲ%���pX�
c��B�)���\"�S�?a�p�^���L����J�b�?���/q2(�������\�%N3-����5��z�Z�֐j���6h�(���B�c�>�r�˝r`s��k��~5#ӣ�թ\{*9y',�?vj]�(�������9
�dO(0E����M~k���9a�ܥȏS����^���� ����%�+�Ã�ģ�G�V���o�2;�}1�<=�&�@ʮ�= m.�8B?�����xx�(�5��r�t0�:r����OAb�����P�E'j�u�*.�u�U֕�@c�e��\%f�/���q�
Ĺ}�r%M;����@�XfVe0]Ԙ�n�՜�2���Y���$�8�\��롃}�Yk�ڟq�}�k���Jk�ƴq���w�Hc&���X�74�&|�[�|?�>>�����b��~��ռ�F5��e�҅s�bO3�GD����]���W>����F[V>|w�tS>��}�%�e� ���h�{���F[�4&��H7V����B����D*�~㜍������!gADQ�z�&����d]AA�L�q9�{���9e��1(�����O����K��fm��(@1�N.�Q~����N��2��&�C�j���Q��8n%��1��Z�b3ɂS�����,do�Pۙ'Ĩ[�f�k\+yo���i3�Ӱ��NJko�Аu� �,��U\8,�,�%rЖ�ІD�g�}�+�U��jZG'�>!P���aJb�F�0��@�2�7e��j�.3�5�-�3k8��E�\�9)E�#I���b�1��Ǖti	��$�_��-R�ApЛ��)�;�m�o�ܽ��1���u=ŋ��#��P?"}�|�Р���jg�F�/7���>1�a�Q��������R@��̺�2|�o85|�\�g%��w��r��ǃ�G����#�nKѝuM��g��!��_��SKt�X,W�RG�
�<�2�V{�R�9z��/eѝ��]gv61�dvqll��3��P��ŗ��1��N'��/��+
xY����,\Ϙ<+���|*�z~��djm�p����f����9��5���}���kȔ��@��N2�����m�s����̵͂�O��R(Iz��|�Q;�MDU/��׸ V�G?�N)�v�f�GU%�Hq]i��?y�^BS	K��}9��8�/P�׸����1���"zWG)R�(��;�����p5U�ڢ�{��]�����p�Y]:�Q���Pa |7�u�Z��<󕭮u/hm�e��n��۳2����0��-,�L���� jX����.puG�ug����'<F&��(�� j�N��p�]`U�u�u�������t��m��x�{�N�T�pu����%��b��Z,yt��t�a����R͜�M�}�c�uM����n�+���H��F���R�@������Ǫ�yb��nQ��?Q��_g����m|�!���u߱Cx��۝�_Sz�nŎ�(�G�,�uY��,7�����B]�mm��_j���Cu)o���N1�5���;Wd��9�K�x��n��FLcNe��$� ���ܭʍ�%��y����J��*J��:t�0�ȵ�	ד�|�.��Yb"X&����A�#���t��"w��K=*���i,�&��9����g���y����օ��N�dل��z1�ƻx��/��ǫ���RHR�V�t�X4ޭ^e%��5ф�N ;Q�`o�Y���Q��(���(�%�̾_i�oUѢ�8%TR�
F��1�N���	:�2KU�UFN�Eg���	tF�B��Ԁ�Q0�)l��pJ�v~i=��o�f�8��e"�}��+���]h�bC0�!\Ϧb_�A����9�I��U����uT�O��46� ��s�IKQB>Ț��Y �y����;�2�K���}kF��[��Z�?wf)���r�.��� Q�Ol���ķvs�t
l'�ze�S�U�|���#�,=3A��R����<#� ·v8�xn��5�����z�H��ԅ?wV7Aty�{��j����ZHB5�|,1���ۃE�p-�m%�A�P�;^+8+�=��w'��O2��#q�������+��e�;���+��_W�\Y��	�c*�'���ajT���)�D6lP�FH)���r2�����g�ZH3�q`�մ
"d<��j�6�5�،�4��K�R��[�00��S�x�ZV[���O�N$|�&�o�PP􊤖���,���f����1k�L%8�6�u�-ފ�c���
���jK�F�R4Cوk�i,��k�?�jt���h劺�o���D�&�~H�eB��r�F;K�L���#!��sG;X;��7l&�|he�{�WM�
����UƷ��]���G�da��V�s��U�YSФ�j��k:�&�g���_�)���>���I�������W-�%�qu)C"yG*�q� �@J���Œһuqy�T.�Rߥ�X|�+�ȝ���~K��(�)��^4wB�Wmn�&��������.��L�d�)����U-��4��v<L�o<��d4�GMt�/R�)GH���������,ڗ4���$��-����VF�J2-|�>_�;<^�ʔ��P��QZ��F��d���M
��s(�
��]�+���%�!�ՇB�)敯���r�ߙ�Gqgr�ho�n�2ET�X:�C��Z�\��e���Z���Tc,k�AV��d:O��gA��6��jtc��yۤ��5Ϝ�˯X�AB�Ϥ����uW$5�"\��DW���;�63Wx�*�\߱c)���c�(�z�
E"�1ˊ���Ä�)�X����h���<-/��SQ��9kN�y��q��*@��(�N�x���!,�MQ�7���.��[�by�i�+�i7i���1^���u�~�\/�g�z����&H������\ �!�.[��H��ֈ���E�T��Ly�1YǨ�����Ӽ"�-��.Х��k^��5��~��Dy��@J�ӡ���d�7CK��\�5cbw6���
Z�{�=��ӥ��^`6�è�@)�@�r8��4�����ؿ��4_C5P,��׏��,э�.��5����8�3�Ip��5�	9�.#���dm"���]�{w���O��o'F�,�o���<I�'�%����~8X^ԠL]��������C���;�y�,`3�F��E9�
;�)�g"<=EV<��.IgN^jF�[+JQHn�G��-��
�.�}C����9
�i�3��l�ZC,����p�+p�R�эj������\@0c�Xm�0+}�X��-9�<=W#����iK�ҷ��ά4��Rp��*7Ưr��SR��j%�=��3r��۞=ܒ��:yvBY��z�+��W��ʊ�4�iLֹyd�S�K_{{��␺�����0g��	
���MAX8b8�ﬦY�s� :�AO-�'$�������}�rꤴ%7(�0�/#]�"N�g,K�ʚ�)m�z�.?�����J5�\�JVUg�p���J�!�$��g��"�B�4{�'�3���̐1;�� �1uq��N�u�����' Aa)4J�n�eA"��>*�B1>
6�+��֝>fE���o4u�0Q7W��?���AP�%�$<I��8�K˜2i���Kt��dn����=��ytQ1�(ԥ(����Q1?�a�T��@��9T_r���C`�eu�Z�R�4�+�BAS���Qv
Gg���!���HB�~�XN���
�-'�p���%\!췬�����l_�ʈ�`&F�h`~"+5u�DxW>�R��}u=�r~�˴�C��^`Y˨x��w���ې������1[�ZY��g��3xB��H/0�Tr����}>�����(N��n��r-G�7I�a�b��-�T7�Q��r���4���^�	��7	��B=^姘���G�S���E����#~v��Z��+:�te��n�\A21,��Az�!AO���5�{׭�^])�#�	��ti� ��6�0d84���; 1�t��5�e�v����j�K�!����n�G��x0�؇�C�\{ P�c%*S<�ɍ@Ǽ��ݑc�8��mmk�ր��c9mZ;��u@�����Jk{��8鯘7�S���="'�8U�$�*\�F>�Ba�L�R�e�#�Ծ�'��e(���u10Z0^�<�M�UBu0c�e�L�$��$ͫˑ��8�Y����������~eNV���+YY�(�� ����r�i+¢
u�d��ɸ.3�?�!��am��{�nX�a�RץN�u�nIn��!F��5j��iV��c�,:�5���[��X�6�L�Kv��"<�N�7�-П�˵Җ��.�޲�6_"4ʙ�6����!��Z��0�����/dI2�Ph/�6Њ�5/�s7��r��Q�끮�m5l��&���(�Ed1���@�9Q4Ac��G@���^?�'�6UlrPS&��E���lդ��X���_�ʚo��K�=�K�I�Č��b)3�C�ێ��K���4�6�Nc���pQ�!�<��7�V��6�n�����m����孵2jLNs�]�!��!t�������Zֆ��X������k���}�p�7��a���e)�}s�P�f2uqM]���I����ɿ�� �S{����$��y�'�{t�v=����^u�å(�X+i5]MgU��Y�$��N��ɣ0{ ׉w�i����j1�ɾF�Gڙ55w���4I���9�*R�L��r���G?����q&�>�ϧV��Б-�ܫ��9�G�;��]UU�}^ym�1�6\5?�-�i|�YC�N�Aۀ��W�8LGc��Prt?tY���:P�����͞*�Hjc�<5.�18�3�ߩ���D�\1�Kg>G���9
@'��N�H��W��|l�9��y������z���=�gY�
T��QCEyl-:������
�A]�z����Vt�%�!�@>��=�^r��� ���i!ʽ��x�-H�8���G����(�g�uYcP�z���@�N�ݥ� 1�����1ެ�efB���!�E�#u?Z��9�ß���vj�L:{		%�ߔo�y���Ϭ���YT
� uеXg���sk��5:\۹��g���A)T�u�1�pZ��\��G-{�l�bD�A�T4 :��7]ׯ%PߙE�8$t>��� \��@%�G��F"�Y�����WO^k���7Yp R �pw��B怐��=��i��Gm�55?�W��q��F{��yG�S������� ��+���K��Z���0�aQ��8ZI*�a(���:��jhL��[��o�C�#�U �|uZڬ6�(��xE��S��֋9�ydѧ�G�����cKg>��&�_[Χ�2O#��܃-S�4�e$�:˺����|�n���8A��a�*��9Qm@���Ơ��,�yc;sOA���_Z�*��Ò�2����H�SU�_2�+A�K����;��8u��f��5A���ܭ�z�d5��8Y{u�}�\uJ!�5������j�0)��'j����bC�jm�UK�²\��J��	 �=�>�{}��K�07u�[�p	���n���O�V��z1��=�>�SW��tB�$�g�9΍��`�Y�Ng]�搅g��%�\M����ѥƴ�w��i�YK*��9�{�mz���w�kQ'%�s��Q����aa+`���*l��ڒ64{�]�8�}��%Uwz�uD|��������7M�� "�R���R��/cXJ��wC����{�e�љ,�s,��g�K8��DQꭖ���U�T��"[��|�H��,]�(ct���^�V ��L��v\�˒����4C:�UH�6l�K�:|>�	��!�,�����kD=W�3���a���8�w���S�.��m��*W��_�Y�_7����Q�3�`�ǅ�Ъ�4Ң[We�X%"z����W�_H����Z�d��-My'�D��f��$˧�3M�1%���`�xkj��7�����"b�`�<O-=��>m$>[*��S�K����D�����{�3��+筅����E����ľ[��~q���O4H�t�+��o��Rj�TeM�h2=sV�)�ᛈ���o2O��I����X��9�P+�ԝ1;gZ�/����{��ǲ��}�Ѐ�?�����\��N=�U�q$i$�oo��P�V��c��NG:������aw�Qn�ڛ�#�T�Kr���5jb��(wи��w��'��#G;��X�Y-����;_�z�d�DAx��k��vTy;9�$r8��R�I�z�f6۝D��b�
�M
�E�C��P&v:����t1C�~A��(��\Oۗ��(���
ѧ�iJ��&w#���ȵ��Ǜ$-<�e꿑ӵ����ԁ��;\�����8��5H���sD^�ssg��t,�9��t1S[� ��Z2a	l����O�1Jǿ"�Xo����ힵ�t�]�����u�~P`�o&��\u��6�n�
6�Kb���}��nZC_�l�GՒ��o��Y)�*��4��2�;SQ��9C�f�ľQ�EBJݠ�7�Wp@��sq��؍m%�j�ǴMo��w����,�'s;�=�����̂�lBkU����N�}')�TӺAg��&���S��S������C��l���s`zx�~6jA���.�/�[9�͢ �[�H)[-ɛw5��]Q��-�Cj�T� K����c��ƋYg؊�A@៴�E�vh~Ż�ޭ���;kX�����,�Ž�u���M�KB�����#x\��AF�Yo�3g���r��Z�I.v�6�7�*�h�����+�{w^�Y�մ0?��Hwu �-�� `���跩�{�ǳ;��|b3-�=��>ָ���6#��Rj��|��,�/��!��&K�/���{�[Il�i�|5��P�q0��NW��.��>��2#��M�0�tH�g�*�2v�ĎiF����-%_���q}_w��!�{+H)�٦3��8���X�w��šC�a/���Q��C	30j�b�d�Sm(X�/X�K�����A�^���r��B+ļ2��^�0�2oH
X�(�?�j��gX����7�߈�z*��S���n�
ʉ��L1�~6�=B3}� �� �G�Ōԙ�~�(��#JI?4o4#j�0�+u�>��,Iwj8�w�;$�@ ȉGw�u�H�}��IZ�{"��?�������{]U)�L}>��?�r���=$7��Ӓt��F��/'�J�X:~��K��{N��@7o�Ϋ�V�g
��!��U69rv�����w@�&|��X��y��P�*��ᝤ۲�넼w6�G'D��7��I��ϸ0wX�{�U6�sZ�R��xZ�~L|�h��54~�ҡ��!	�Yȗ4�P�')u�7����o�#�:=�@М֗U�̿�t��a����E:A���P���uPx�Jc��6ޜ�A���u���Q��Q{�
K�%-�ų�,�fF�|t W.�+3*�h��o�BBD�k�U(��&Zc����!�(`1�X��3�N�9�����r˜�E�����_z8�R�����6n���T��O�!o�F�_��r�Q	?�5�=:��l]	�N��̌y��$?d�'!Q����%�$��ZC�4I�DbJٚ"e���4�&�!{�Ǿ�03�e��6���}�s�}g����{�=�<��,�v�ˬ�p�b�jY�B���g�F�����3�l��lJ����Sc�Ĵ�ʫ��Ps|������w����<�+S"��V��f,&k�P=��\r�NT'�O��	�����3�	��z̶ܹX�%@]k�q�
�\��Ȯt�ў|!X�R��&W����LHV�)��g�29�`MK�i�vƞC�C�����^e�@%T�T�W�*�d}�+u���V#)�C�빪]�Re�C�ۊ����16!x'��m�~�ǱO%��٤Π�jù���:uՓ�b�z%���m�9*��ta�Ms�E\���w��sC��¦M��^){��\=�]���l�O��B����]���XLC4�k+�{�p�zHۘ[��Ҍ��5�Ӳ6?zH���!T
�|��a\@�ϖ��78F �۾"PE����`H���94�t8���L�rgH6���������E�e\y�E�#xJ�5!��RaNy0�(4�p�o�~]YT�ԽtY *��^�*�w�g�K0���r�p�Ğ��d�m�!1}�����Hʨ{�Ӈ��J�YCY	�vٯ�aM�
L��+�RF Ae\g��S��h�n:�<k~�u��ϣ���h�7���*��o�	�K�r���ESӾ1W;��7L"��	#͈�"'I���pO����-�w������µ��g����d�9�(ԯKlu��kqA�Zj�g��G���Ř�K;��g�U��;���$'�ސ=����I�tEҾ���0JD�9��s��8�R.U@a��Y�#�Rs+�r�&�΃m�:���P\�A�WX��bP�,J��)[�.����F��P�N��/���`���]l�=2_�*DOw{/p���7�M9����S���*�/��8V�lY���,��4
P��d���+�^6T�J6�W�����"!�_�q]��u�����~�����t����?}{ࢵ��4+)8��e�L4�I�h]q�i"=T��v��D�Ԕ���u�����KAl���p���:�]jZ��΂N�W�����ʦ �d���^��~���wA,|��,�rKS<�����~˲�^fȃܥ�=M�2cҙu�[�ʣwV�$E[�M��ܮH���c��)��(�1K(����^��0��2])�2�b�?T���#״��9|w_F�8A3������W�,!������_�����q'�-��-A��؈<���#(��JO�� �m캍ը����(6��B��J��Fߒs��.���k�I�� Ty��0�2:P��%,�G<;��X'���I��6Z�*?۠B��o��ڵ�(�5�R%��~գ�Wh��ֈ=+T_B �Vz�%,U��"Aiq��А�bv�sr��F�wp�ݽ��ȥ�&�������T!+�f���QJ�2~O�w�ȗ�4��w��Z���YR�8 ��]>��V��Ω}z��(�.D��8��T�fn@��z�zc�j:�ü	��K�lՇ(�2���t���[�ʁȢ��K���8oi������J�ph)��O	�q��S�P^X[����;�p�~M]b8�f����PGC pZu��P�ͷ|�N�������~Fߖ�ρR+Q��k�T��J�i�L��J,w�P� F�ܱ��4�Ҿ)��Ju�S-^�����oj���ЧA��(�l�}��~���������U�e�g%�}[���f�Κ��2ڒ�L
��T���/�d�6j�5�Bu+�{�iі��n��E�K�k˒u0^��Gџ�	�tT4R�;�����42�Y�NW$`��@�j���C�� y�J:A謾 l��҃��A�^��QSRzPE�1����2��TL�hc5f�*!.%튔�ީQ�jS�B��ي]#"�,��G	NmNV[����]����C��r�K|:]�Z��M���'����l��Я��)1')�\>��Ne �����2�s�f��C�9��`]�闓�to^:�^�����:^w{� ���*���j��E!��d#�o���z�{�o�l�򯞣 ���z@��_&T+>�S�U����BG�"�io������>;Jp���7�|ǩGZ�����k��\E�&ީ�P�g�!�R��.c�Pz&`Y�e�Q5K�^�ƍ�\+��,�;���>:�<�m������#r��/�M=�U�׽S����E3��	6��	��A{��D�'>�Io�#;)[�F�2)d-�t�p1�dl�U��f�	����BrsBZs����^�9����Ж�d��ٓt�˖s���J�~\2�c��ܧ���7�;%��N�Lw��Q�7sW���I�U]��h���M �E�d�[t4�����`/Q����䟡��j����PcLY�Y�ܪ�Q]��t��Vf���M��3���QT:.e�'g�o�4f|O�!R3�r�����[� �D�k���dX��ט�NQw�]�f���5�ᵩ��G8p�/+�~���l���ߥ�`C����a9�]lP��A ��K���ϜN�/_���$�[�[������]���7���ρ�}��1����~���r-��.K��������T���vlZ����h�꧄{9�#f[���hڂ�E��d> 
���_�����Z��o��E�F�Ǳ<� >��f
��0!�)�[�'���B3��h�cu��ye��)��#�,���@ ��T�}��uP��];6�R]���ʁ�b� BI�`��W�U�4~c�z�O���[�]oP3{@�*]�8�ۊ`������_4)Z-,������=����34�$�2�}�O���b�"_f8,�#��#�79<^�Ą��8*/���/Db���9щ��(��J~�=��A1��� EC'w��*��m�����X_�}���k��(kz���7([_�2ry�K��K7�n�$��e�#���CZ���e��y�`B��[JQs~�DyW��S��2kn.8�"��6n�	/�����͒6�F��K4�a?��	 �v3��O��B�2��DC�sW�&�إJub'�(]1N�lm��{(����FAZ��9�C1�eǏT�O�Ȯoz���^5Y2A�AK���.�����@ގ�ǈJ5o(� ��nä�h�B�dS.�cx���v��CV�\.�)H����(�S�oz3���a��Wٔ�t)\(���39�j�y�kp�D��������m�n��i�c�ܲ�f:��^莡 �3�>0�+k�؟9�#�9���@X����h�#~|�@5A����7�iQ<����Ff���$���l]=$�ۊD�Zj]dzy�QZɢsvmYS��_ʊ����$��b�{?�RE� B}A�Q�(��Nʽ8�n|�@`��.	D���!p��ä��ߤ���K�]�{�_G*֡��=C
��'�X8#TA�8��ɏ���c��������L����H��!��Je	���K�!�;�H��u�a��uҳ̣i.���!(<Q�+ǰ}z���;P��t�v�oR��̨�i%�J��v#uǢ�gy6��C+gj����{���@�{=�
ٕ��ib��Q��o�zj��`~K� ��Ӭ�+�ٔ�0y :M��0a�RU�!C<v�����Ԟ���b,-d����wE�|���^��lʶ��	�XB��.H2٫��#��>��? �H���׋�|+\F�U�����)��W) �B� �F�?e�����8W.��$�G�'���f�1�B�*���ا/^��,�3��	�M�xv3��9E�(d-�}�&�aE�8b����&��T�ĩ.���7�i^{J�E[��cŒ����X7��l����T�*V1��t��!;Q�xM��i�'O��S�/��E���/Q�0i�֋:R>^�-N*��y`�+��לum�R��2?��#]?�[�24���
�ΐٶ�E��2|�i	 nd�|�h!�Ar��$�E�֑��_���Q�&A5�~י2ĸ���q�D�}k���=rm���G�����qUh� �e�p�s�ta/f���82^��|����N]���C���T0I��>�M��x|E�_o��U��{J�L���U�*���X�fG�G�~M��SU��IB��R�w�S����q��Ą�� qX^�Kq+tM�Mf���8[�[&�~���Cg�[�4C(4Q�t���d��ť�HS��W����_C���T+��ţ�a�.�C����ՕAk�͆�"��j�X��t~�A�����	���~���B?�9�H<���_AUSQ���mP��B��!M��BQ�<��S����E���A6�3���|-���?Q:�����jsv��#6��ULkx��?�F�i��(:�yoyyʙ�tJ�`��ۼ�7�Ι��x>�p�{�hJi���[R!XS������2�:��58���M|r8��GU��֛⁭@�>��;[�����cu����C��C�q;��J�Ŷ��*����������ߋ�!�-�!Oߠ�\��j�8Z0 )�̘���V���z�R���rt�-�.+sC���x����{3:ίKHIf*P����`	~(�Pm9֊a%����+���ARP���m�?��dC�|�6Bu�M
����cU�,]�;�
�����2;�:��A=Z�m����3Cf�+7��K�.N�u��b���<��b}~���˕	K/u��/������Xβ��`��H�r�wC ��B���<F��n�4��r�Խ=����P��3-X���ü�z��Y6q)9��$�ZT�X����їE~�K���!:�S�D#ű$�s �q,�����͓��/�ⷶK�T���������?��~��!����#�H����Ѝ�<�w+����y_M���	�>3)�����>�=ͳb��'����%����'�1�@0 ?biw�!��ā��f�*"�	�����)]����*֗��՗\��$��ll����5<�-%�WMf���X��Z4XT�����������Sy��ݸ.��:+���i%�����-�����&�K�ն��U������S�;����Q���5��%=�l��mz�1H���5r�jy�5�B��Į�i�,��W�w]����Y�DW�@�U&zr��e��͔Px%.9`v�8���/:>�q=~ٟ�=�v�/�*��H����ֳ�y����kO��m" c�H��oaȃ�D^~���[b��e�P���4�++�\٨�@��)�pBb����ȟ�!Ǚ'���i���G� QrHj+f ��$g���,)��1�7���:� �EъCp�j�p�
��J��؜Z	��<�m�"ĸ�ы�o����9�AYڌ]䗟a�w�h��.C>hk�9��BD���^rv��-�9�-�����c�Q�,^1�V��IםR�p�,���M�>�~�V�`�j��N�S��F9���K�⠙R�k�ޤ(����<�AYZ�b! ��[A6Vĩ}�>,�ߴ׷��$��{쇦��-,�����0����m#�BHG�����_��:Χr���S<�C;�NL��&�Fڲt5����G�:��RqV<;��\�p���5iIh���3G^ҡ��O����]���ӱ����Dr�V|]b�r�n��	��M���c��Q�)��lx8xC�ra+AH҆�g%9uq]�[픺ҿ��t ��h�zY}�I&@a*������Lr�,�<9�$a������&�T��u">Y&?�y2���9 Bj�݃��t��Tѕ+#ɱљ1��H�U��Y�%T�z8z"9�#���+�/Њ�Y�9�*?#V��g��I�8YP���+f]Z���4Y�J|ZI:��f#��D׎��t]w6z0�Ϟ%1�P*"jW�#�{���9
��biz�M�����(Xs���r������H��g�Y���~�v>-� �6�e?IWu��Y3�Sޤ�r�ͅCj5C+�Ŭ ��z��	��Ԍ>/�+�[����B��i�Kr$čKȠO��e.�ȅp������W?�W�����l5x$�*P^S<������v��Sj�YI�M���a0m�=q.վ��Ub�H�V�f�7�J¾�������w�pUWEs�1aV�:e1NZ���8����+�%j�H�����\��^Y��(!�͛ଔ����^8 �v�k�0kVJI��A��U��RSLcimٕ�G�o5o��g�ݺ�u���a���ٚ�q��r�b�#/��5�
�U|V�0a7��)�ښ*����e<K���|��z��n3EĠS�,z���t��V�X_U��l2lHG��%Z�t�Y�BN2��+�XD�J��<����t�i�ͱ?�������,� E=�>��=���!��[Z�Hւ?��~�*���˕�삵��<,L���N�0�v�ɔnI�F&EC(��gu�|zh��Nb��R�騝��w��^!���1��D�6�6O���x�l��B
�j{v/���1���Vw�ZG�k��҅2j�%�<�� U�.w���2�[N&I�O*,U�aX���4.V�}��5�������֞�g���-W&��I�G��r&�L�̆�S�q?Kue��.�s0B���L\�a�r��yo�z�=���;����<r�^P�D,�q*�R��(6 �D-��:��-�f�΁B��<���};��������Vj3-�!��d�n>��i�����Xb�:-;h�ݴ���0��Z�ꌫL����m$x��ҝW�ު�������L��������{�k�ղ2T�gjkYb�9ߙ<�Ε����$�#F�����P&�[z+��p*�w�-ȅf�",��+>�ler������Qp���;Cu��,;�G�f�Z��vv��`g'X��3�ԟ���%����g�PОe`�����;C=�!(���&%�G?�/:o(|�}ʪ�7t� �2�@(V���0��?m�P=���"���^k��Ĕd�������oИ�n�lJ��46�<�?o�8ЫA�CioS'%f�B���2���5�M���A���7f��!G�Ϳ?��#�����*����_1~5on�6
�a�:�5X���/So���0X� ����pBw��u��q�㝾�����)Ŝ�q�.�4^��<%��Vt6j���j�4�[�qJ�<p7=	�bk<h���<�Ӹ~���b*n��
927����^�š*g;c4Xb�9�0FϞFq0\p�&��&2��B�Yr޼5�
ӎ�#��~(��;N�'M3�i!�My$��g4���90a�jYP�Q�N&�W���j3~�mTp� TKӲ]�׵_o��9Qp���F�R�u�,@���޾��Iu�_�Ϻ���';k4�'�Ǒ>�K6 ���i��:"�?,Veu���@Qm�2T��mT{z�̗4����|���#����h�*�.1�&:⎙tl��&�v��.�{*�h�u���
����)^T�G��ޟ�$�R͋�4���qN��ȡķG��@��F!<�JAO�5G�Du��ЊHl�t�R�g4��+V���J�yݫ.g��ױ��n};�C&��v��T��Ʒ������2m�����B��F�P��<��:�^���&��Jx�e;���ߎt�]C����^9����g��Z���WΏ��k�P���~X��O[�Ͼ�KYuj~���2x��ǩq@w�����)�8ȣ�i�(�zp.9pm=m��*����4COU���W�z��e�\�@�9k,7�Ǉ#Ԁ��Fu��R�9D��[,�BhJ5�I�)](�dΪ��pI��UfJ�\VJ>(L�NJ���x=o������J*�ֲ���,�B6a��C���/�]W��� =�k:N��K����ᵘV���]��5Y�x�r��=�w����z��m����?R�ϗ�k���������C��,PuGEk�ꆖ���ŝ���A-�Y�"
������nM�OIW���ygb+J��b����TX�BU|8�=!�,%&�3��_���\�K�6��	��xe�?P��ik����՜�ߋ�ShX�"�_ܮX�1���@�c�,:Fɇ%(��+�D�O�J�j����I�ʙ�7�W����ٺ}��N	` I�9˫+����X�:�`�DJ��� ��7%�Q(��g:"�]E�C@��Mo�U�Gɽ��C��6p� ���Ͼ�\����X�ا|���]!g��\���XB�X�-���Zل�:˫�=%�O}
{j���� `�����b��:I����ĵ�	ݔHnÛ�=C�o{��^?�-�Ulʛr\B��Ũ���øBkL�H�擛����kH$�#o'I�8�!�啄�Є�C�� 6�`�?�@�1��X�KI��$�6���,K�DS0|�(I��I�;S���MS�%]��E�N���!�SL�|9�5�J0/�w%�F�痜M�!���`��[g�̨6$m9uu���rҎ�O�@U8i"k�\���NvI�8����H���i��SU�j���1��:\�(M��V�k��@^#����-�����]���Ҍ&3�b`�/������l������ ��cn��U{�ñ���:���< Z��S���p�Q)HT���>@(�� �kL�5���dyn�{o9>���kY�7�":�xWH*u����f!�L � y�qpE!����0�e��� 4\����{J${(����K��%���x���W���^��ruw'�qҡ����}�<=�i�tV�Rq=Pz�J/z�9�#s����/,����g3�Y��TN���8VL�AS�` �*�J�)%v��1�6�w��Rn�TV���E�(�+1:�j��Aڈ���^����Y�P�:Rum����A>X`�!��x�F�L�E�x8)�&-c�@,g�)-u�AT�i�TQVUI����u�i�	�[�2���Ι*���X	�����9��K$E���)�P˚wv�z-*ѹSc�j9��~�XU�a(�S��hV?q�G��E��'�6��w�����L�y����@	��:u����H����|��4 =B�P�VO�BW�.V�Kȩb��s��\Y#�Cmd�Vĕ�W�ظ�T|< �����]���]jW �%�C��w�AMTW�Mt��t��G�Xb�~J�N� ����荂���>�|V����v��&S���keR���/���x���45|p#[?��,� ]��]B&�m*TI�@6�3)�(�;O�X'C�����d�W�QBBn�۔����$�A�>�I��Oܙ�=G��9�(�)�N��n@�����`uZ���{�b�	vh�|$Q-����̆�D	~m��bV0g�gy��R-��k�Z��L؉�8�eh�v�Z�ml�ߖJ0U��%.ղ���4�8�%���|)��,Q�����ZRf`���Ѡz}I��:C�ۂ��Y3y�v�lAt~�K��� �+�xd�k�N3����ygr���\a]^*�3��׹�DP��U��z��+>�$�8D�X�B����� �3ٝZu���MW�ӂ�G�
,9�Ge���2fn�vV���^�Ms�� �fԲ�S��c�a=
���ɂ��}W�>l�7�z��"��%�5���/��C^xH���g�����?���	5,K�B� ��U����_���6:E���݅��U��j�x����OyW[���g�c多䀷�p��Yo7�OO������{�D���4��� �\|�晲Rɩk��r��hF �Pe)����^.z��H����[�1�B���?*#��WXD�uh;7�WZ+�A���O�<��Y5��#���%6M�`;$ot�B�qr��䔘��t�~W�^�w��~�n;�S\D��:��-\��%�/��R^�`��g�IB(9�����W�&EOP�X��+%�ّca��=IVd��k^�N����qs�D��!r��גC���̣4��q��öD����SӺ�T]�i�=���Yx��a勡��������^H�{��^=� t�<�Oj��*$��C���։�UQ��������r�n�|k�
½�m\GӪ�TO�m�[�[P!�@�jNK٤��xj��C"�cݚ�,���\�c���l�~�iB�?8}"�+����ўGX��ϓ�rBq��~�6�e'ƽD���1�AYj������K��MİG>h��9�3S����㬗b͡�b8Y\��̾yT�8{t5l��G�P�j��嗋��[NW�"(Z
�����u��8�Xm�#��R���|��eLIjxJkJ�ˁ���q8b)�r�%����ڔ�^��"�p)�EG��'.�8�$�9�(u�|ϝQ�PF�7�8Cs�,�4������N�Zx=�S܉�)�o�*��h�9YF.]�|ɀ�s��>bS�A�6�d�.�گ�O��;㠝JN`9j��t،H�:�7�a��-�B{w+�󪤽f�"z�[��Y��=Н�/O�13���,�)Fҏ����&��]C�A��ȩ��7���J�U�S	C�Ժ��Rehd{�q�Z\^�%��:�g��=8^Y�d��Є@qB9���ni�a��x��e]R|A��t�6��@O���v5����P�B����<�>Ɛz��j7p�D=O1uhr��^��uG��S�q�d�9��P�v,[�		��SKy��y�����a	�
�AS�Ћ��B�o�-Y��v���&��d+U`�
�(�g��+�'�4�P������W�2�;^U�pn��5\����(�e���Rd(	�I���p����rm��Z	X�W ���z"3�������l�P��N��Ѓ�7��nB�����&}�:�3/��u��˕�&��=6��8z\L`8������9%s̎��`�Z�Jp�;��<�������	]3=�y����څ�Y)�=M���`&�_�Dh5�)|�����"<�������_�P_�r�95����D[%� x%���c�����H�{�"x�f��%`��һ�֕.�o��M|<�<%�<�=�S
+R&ۂT۽`V�t�������kīD�������h�}��K��V]��B�}C�h�Qc��*�G�.�fCUXe$����vl}���e.�����'V����Ӝ��u�pSx��ÊO�Im h��+yt(i3�LԆI���)F��/0asOIɕ�W��','$�R6O՟MY\�b�a!0*~�.[yS[�ix�^��O�T�T��F݄�����F�^���mu���+H�7�,.i�,3���2Y��[x��s�	 �z�a^���y=�l����U;|c�	x0��JW��A����n}3;�[�a@�x��|�6�����xE��vȻj��wrR?�q��h��ܾ6Vs7CU��]�V�~[Ek��f�	���T�p��u�=$��ce�^�o�I��ȉ��H��!B�0n��a����J,=�j��pAͼ'�;�����
�L#���)���;��9��ph�mW�P���z��Z������G1T��k����HB�������.N�F��W���D�F>48$�j=�d1�J8N)&u#�9� ��t BB+�Ϣ_�z�ם����+mQ��띩���fy:9UIl��&�c��-��kH]� 2( O�h��@�i,g�Ƅ���p��<���w�5˘�A��뇐B�i���jٝJ�Q�1���W�����ј9n�|�߽C� H�@XJ 9��� o�of��:��Ĕ���
��pU!G���g��:J�|��HRD�@/�kw�ʁLN�A�����݌��Nr:輤���{����)&8�� Ua�\�O"���TG�Ȳ�0��ukN�F���x{dy���A�gV`8]��Hs��~�}���?���gI��gI&��DB���&�w�v`W���G�S��������1_�jb�uT��AL�y�?�6�0���NR�HO;�i ��{��=�P��]4���#�6�.�q���FǑ9ɅO$�1����� S"���ш�AuHv�5=W�~i�e�ev�JZ���)&F��
�{Ǟ昅�,��'1��ь���	���{�8u�
9�a.* �~
��/nK�U��i$ql�4R7�5QO�|}��;k�,U����N|\�1�j�-B��h_�����~����5�4���P�y�7��Q��)���w�~�_<L� Urg0u�'��0��A/�	�,��B�	K̋a�F��JP+]%�����X?"�o,�ٙVaj�W"��㪥h��6�6������TM9hjc�Nv���C�H'd�Sxn�YY4_O� rN���.���ӓ�u�Q<j"��w��`z�G���>�c���/c�H03��f����4��������X��������k�Cd3q�_�R��hN������:-i)}�͈~����WCU�a/��J���牾�ԉK��0�Q���}T�-��At�CO}����uTg�4aP[����`P�Q-�Pװh륫�sTf�����U���f����E��p�'�u�pD��@���B�H�\��j4l�bAxŗU�u����dA����LӍ=��pk3�08�ڠ@M��}Nsb�:������_��-�\�Ɩ�B��`
KF�e1�V�T.��N�=�f`��tZ����0��*U\^��]1�݁%�#��L�rZ�GL�h��u2"������.� �l�]����Ł�ZH��3fSM�L4��o��#�*�/0Պ�J����q�,�f��'�.�G"G��1��7�,�e����/��,ڄN�^s%����ZT�'˰�Wk��Nؿ "6vM�TځYy�[�0V�9*ȗ˳��Y=)	���2w��A��1+�e����� W��� ����2�u���	tT<�Z�m��|i�x��q@X������|�/K̓NK ⱔ�E��:Cw���0�Q�t����$�*3�����Յk������ӵ��{S�xK�� 5|׌�to�IB;"�R]ȱ�ܴ4�X�����0Y�GH������=ά�PV5���<ƒ�ޣ����\6y�3"�1���d��Fk*GB;�ӫS�(���^%vB[�K� �C��V4$����$y�g�K�7�GFPeThZ8�Z8�-FM]b;t�ʠ�5��L��!�~�c4���o�B��e�`U�Dc�X8�� W_xY�&6�ݸ}�Gq���<��|A��@m�T�Lq��[�A��ȩ�^i�S:�MI�o��H����Ѯľ�>�W�L����#x�Ģ��)X9�gE�lkn�|m/r�DtP�-|�p�( ����]aY4۶4x�uz��"��h5>����O@�iUQ�~��4��6�9�O� �\"f�'��g�S�~3Kx���K�~as�S�/����;on9��.	g���"X���϶���N����O�n�\O����s��w;�����h��I��X��^k�'�O�xfpc�a?x^��=��>�<�9��U���녎���{j���фb���4�q9�O9����Â�I!�m�pw>)�Wk�՗���������'�p�T�b�mT��zD�3��ju,ej4�v��aD
t�x�M>ѫ���B�@��L%+�u�$]���u��2�Š�&����|OtP*TҫĎ�N�#E��9��S\�	�j�=�yeݰ���<�_��U*�@{8i"�����ǎ�,��)�f �~�Ԡ�<D>��^T�X����8^�{_���ȞT��EƓ��.�4�f���S�?���IAEt54js�&���f�HȰ!	֟��ö�O��Z�p��h�5]��T�Q���6��H���͙����zg��!�Y�sS�X(X�)o�|���� �\��%�w�a%��	C��I��4�Lp}��3:���D/�$�J���702�;����רs���֡��?����`���-�/��C݄S�5�4-.븵:ԧ+h��?��@w1\ZSYy��ř.ù�t�{��V�|B�YF���7�T����z@^��g5i\"��؇�B�=�M����%T�Dشqx����Mw��q��,�ǏC���0�/(��aj��E�T�٥��S�u�r@�Y؂ܝKXfyC�b�*?����/_V�#Ɩ�Rm����J�~�NA��0g&���z����C�*��Ÿ�_�{Ɨd�=��������}�A��O.���m#g�� $���w���3GY�Ѷ��;Oy�Fu��#��6��j��8Q��{��۲8P�]�ytx��� Q#u���Y���'$�Ʊ	�Y�aC�c�/�9L����#���P�
m�������:Z7�7΄d՟��MY�X�s�����vno�!趓��%�����B�)��%�d=����&��̄;L�b�GR2�`�ǊE��O��=���s��59��
�Ẏ��*�ǩgPW=�,���麺.��<��G�5�B>��ā��r�-�^���w�Yd�P|7�6bq�#b�Bh���B�$ƫ�)�h���+K0מ\���V�Nq����<9���ԣT*�Th>�=���	!�OJYP�Ҍv�Uc��B��Ƒp�L5�J�=(y�\�x�ý\
/����I���U�L����@ye)��0\��fUx���E6�ɢ�U6�1�I��_���o��T�]�K6v�SA{X�C$]I���;��F��@�ذ"g�)��o��/�=�&-3���+Z��U)�f̦��*��0N� ��q+���\���b:�S��"�����ڃ�%a���}�UY�֛���lt�jK\?���0�����;�,'i�}�ѧ���oD낺��J:�S�~5��G��Q�� K�9�}b������GC�pďSds���˼4kJw˚=Ev��/B��
8oy&(܂Do�aS,��]�Hz�e~׻IN�)�Yee]#�ʹ��U��_�79#��zF>�Y���J���5B�T��tu���	P=��i�t��!T��Gy���>�e�� U�s����^�����!�k,Kڿ���[=Tg���pG7�)��� �b�� U��e�31�T�P͔l�����XҮ�mr�s�ן�J#��ƿ�{r�v�P��|��������y�s����y�(�ǙhP�>O�[+�#��7�E@�`{m��׬4{o��ſ)A6�����̀�aԠ�O�:UP�����E�x����"k,�i�YD��@~+Z�L������7��3��S[Ƒz���Ы(8���!�r�t��� �L�2�{��Le��R��P1�(G��Jk��6�A]�X������k�̆<s���e(��֌�V�N���B|�	���K\���@���
������*#*)@�sKy�~
��F�9�R��$�b�K��<�j|���%}����SV��n�Z2%"�������A��w2��6?�#!���\�R�5g��;�)�X����dۢ���xA����T�F��Н�%����j��wo��֬�b����hKB�A���g�X�ݢ])-��8W��R����D��q��*@��|�/� �Qp�7�P�_A~���')++�F���� �{���]�U��BMo{�ުe1ڍ�%�q�����>X]�:���/b'j� z �>�:��A���tF��;i/�&H�e.�w6� w�w��E�aT�bLu#f/�O�\T:�P��:�e�T�E%Nq8T3�I�*jU��Lܹ�K��ZSl�=ϔe���h��R�ƣ��������r� �A�ǂjTsz�5kd]"��)�
Uى.@3-i�$�՟��mX2�.@S�wJ�d���X���֠u��KP}1�.��g'%�J3U�|o�vK�H�L���ѷزD��gji��?���/��r������R]�,�+��\����*҃��qkX�8cWr��\�e���Ù�u���8��j�?D��?��Fb��&�B`8����N��0Ty ����<t=�bآ���*���^�¶�ߟ�hJ�3�Ws���ò���s�wɀ,�$Ե�B����PҮHyԜP����*sKꌈ.pU1�z,O�c�]���Ym�+�S����u�9(Б���\T�ԕ1�8Tnep�R}~Z݃�8�UmW6��y�5�q6	N�4ڴ
�S�n�STӥ��kG�Z�]7(+>uo�T���K�m�X�S��� ���@��%�Iv������a����:���� ��ǧ^�P�)A �����\�:�=%�`㞻��HtDq�P3=35�DAv (��C�V �Կo
G��G�/|TM[0���"3؎̅��=�X�U��f��~���"S
�N#7��虳z'�O���Ć e5O��G�@fɂ.�/ܲH��(� =��	�$�j�����LdV�mI���{����k:~��3�xl��Ԑ��uW`T�3ڒ��|��_�iM����]ފ���-�����=I��nY$�����d�G��쨫�Z�D��2��Օ4�S
S�c�i�%}���d�0���$��"}�ޢ��ܜ�ܑ���{kC\��R]�Mv�k�)�BN�+�6�t
�5�շ�B�c��xsV��-t\��*���8���oIs R㝢����
rϤ��_��:Ƃ�"_�zt����we���~��i�"zf�7$g�<�鰮G��Z�Z�9����r"{O��Òj����ՇJU�
Af���КY�g��0��%�y�#zS�L��x��V���4�~�%C��戣R�J�����R�L�N���v��Xuq�G@�:�'<�S���s����gAd����f+�"��,c��8�'��̎�7XcBʁ�F����\ی��1�� ����ۦ�c�Ly �9�WM�g�yN�
[���Gټc��貆��"i���
�B�#TK�l��R;��X!�]��啪b!QY�]�۠��lt����:�Fp`�����ox�k�R����dC��c���{	��Y��$�X)��7]��Z<�H�ů�]$6=N�#��.�����*ZLtw�[���) ~��*;�w�.�t��+d�Coc��E؊�˧;�-���It���2��6�G.?eh���rD�fȹs����a^Svmp\�]+i�bY[}�fAh3k�&8�Fжt#^0�!���*nF+R$�'mW�1�AR�����9Sj�I+�1W ������K�j�F�z��!+��@L��u�
X.:Ɲ� ~�ա�Dݜ�?��^3K��&�L���(a��a��)�;���@��Nv�����;���g�,t��!��8ǜ�(����*=d��kF�);g\t��'��� ���x�o�fh�8��t�(��6�mr
Ց~Ӕ���6����m�.�v���r��Bz/Ɔ@)�no|i��'�(�俪������2
oo��ʢx �4��ad�C����`�H���"WV��պ2�T7�)���s�"YS�g2ȚEG<SУ)�����f�R
"��qiWDEW�n��7��,�U$6e�Kj\ɿh�Q�_!eU�j�+�\���9�֌|�1b�[{�
��HJ#'V��ϢX����T��Q~I}�p"�b��r�(U�%ڈ#R�-�T���d�n]?|� � �7���*>���~>�BI�����d���'���3��C5Ѳt<g%.�qM3��NOؕ�w8K��2�P�{��D�AIR��%��s7�'tXF��]8ئ��05C���6[�M�
�����o�=��g�Vh������`m<M0�.���-��~w�n�i���}�z�M�wU�_�%�%p��S2�V��y+<��*�E���_w��	Y8x�P�fꝙF�j��.n@O�|�0�������ÃUzt�恂}���馩� LpY�mNu8���7�Y���Y>�|�Gr��@�|���|�[��?'<��aO�ZW�N��"�KѾ�4X��z�Xt7]�M����t��$��������1&�]�r��M���_WD�ä�v�JPZap�#�����u�c�S���p�Av���[�2� �rs��_�]����̴we��{A�ϥi�P^*a��sd��t�'*e}�T��Dҫ)��4�0+�{[?-h�`x��2e}/د�'��?SZG��'�NG3��,n+.��(���sO(��۷׭`%���C�WtNEb�X@3O@��C�[�eT�\�(�E��-+�.��0�]��A�����)��M��J�i��
!rY!ZM�W�`�Z�(�t�1\��S�E4�������9��+�K��Y�*v���0�I�ôݲ��V���_��)*��5T�X*W�]��j�'��5"�_�^��z��'����Cd��v
aQ�G���n��"8N��7$+�c��p<<��i��h�k莁��s����"E�+�;0���yR�S�tb�YM�/�g������mPǽ6fA�w��Y1~	�aWB�6���*�M��`��Hg��A`5��A����9S�":P=�G���}I���T����Tq��l��s�R�������a�L�߷�8��||r8�|F6AT^�[<��i��F�.�ڒ�B�/E��9��T������g/�vb���6١�y�`�Rm86�.��X����D;���	���lS�(�{���J59Q�f����-c�^�p��,��D�$:F��ن��қ3�:��R=�G=1��}�R4{�(�?Jm$���̈́��k%�mlA���g�+�PR6�h�~�3���t�g�!��]�n}�&�9n�/ձ��c�	oל�c(KF0�7����&�@�H��	��Uc�P�д���O.�/$���3I�x��.X4R4%@�g�
�T�pf�ȷ_QU]cy����%Ϧ���8��|��B:�B��dk����,"�ၒ���8�K�_ukfu��i�k����u�|����NZD#fg��ez@~��Gp
~���sU	c�&T>F3Z�*g|E��(G!gP��`r���M|�t���~�EP7u���U̢ee�93)��w��a:�Q�zNnI�`��Cg�t����fa�F�	rD"�=���qCg�y2���'�N�2��`;c}X_B�L��:�I����EQ�[�S9x�d>�0@&aJZ�^�j����ghk�N�Ym�*�3y%3;b��.7C���d��P²I����&�2��LOV�kn2�Y�g`/���{���6}u�e�F��^Y@RO�����'���92 ϨN�Z ��RC�����Χ�!�S���m��0T���&���fhY�l`���q>P������6��A���t������H��S�j{�5w����p븀tH��޾~��>�uD�+B]W��8ay%tΕ���ʂ��a���Љ�(�E��	:S�)���f�Ĝ����(掤^�wk�5�|�����p��׆�2Q��D��MI��(���[����է�N������FW���C���jʒy
w/f���k��g	��0z�SlN�� ��1��7�"t([j8�t����C���nD�r�w���mL��k����s�T���95 5<�D��j�D(�4��כ4=/���n�5�	Z�.��L
�oHX�>l��/Ln�vÄ*]ٿ�t���[�$��yo`!�m� 3	ELN�E�B�T�X�WBK��5��[�>���S���X��g��F�1��E��B/r#����}�g=�F��v����K$�71������� |�lR�C#=�����@#���}=93L<vk��|y��Sһ���ys���\Dͣ<}O`%!�g�뢼yA��A�iB5��W��)�*d�?����3�H/:�)��ȡy�����m8ߌ����Cn����)K8�q��u�s��
���܇��p��2�/�X�<�G6'�"sd�:��:�M>(K�Q��r��d��1���I��E	�|�lsDf��i2�p���q��m`z01��9�œ��a�+F�@[
=�����9�G��I�K5��e�p'�E3)�ę5(pO�A�`�p�����83X�J�2�3���N����6"Jb�jD�� tЍa�\���~�ن������0X��]���^�,i�6M:�y��,*r�v����FV ���8����E�����攗�T�4� �=��d��M���WdZ8dj�D�=J��2�
�S�Y��C�_ɕ�RݗA�E�E��&*�6}��H�"
�@��t�4���$�pp=��o��ˌR��WٓNҤo1�}P��h�I'ws#�(v4�{�ޢ�ka7������~ڟƃt���0\`�8�&�>��x�~���܉P�R\��[%�v���[|�C"v�(�;B=[i߮��s;������h3�v	�C��#;2���jR�x������9���&6�(E)��@��H�����v�1z#���/㋤�t�P�F̌�!~J�`F��ۖ����ۿ�-�Q�(*��]Q{��hh��BQQ��-��4Q��vDE� Q}�@��)`��K˝��n(F���#j`���p:�[JEƝ�]�kq!�odt&��&)�QG(��	�"�ٱ��&�T�c2�`����P��Q�cۣAD��6�ɼ��&6�E�#�E�E��(�Nm}�\&�mֲ�(_���i�?�i�����0ƥz��9��y/�$��=���''��1����n^R�O�P�N�������ho������s�f�42'*�"�A�Z���cEtԠ2��NS�j�շ�����|g6i�g�]��� ���e��x�E.�L��jQ��K�\�&�����U��w͈�����d:�6r=W˂��TM��ө���/Ƕ�`��q�Z;~�"sA5�2�2nk�c�߃'���ťe[-����˪�{<p�p�-r��.�1�����r\HW���,�K�єť�4F���Nvi[�[/�v�6"�7r)�
��Kt袴x0�
������L1>�1��w�v��t�ۧ:^�ʽ��{:W��O'�����Y��[�ؒ
�	�0��/�
T%kZ��Y�3�)�
��_�Q�\�S�'���}ç6����Cu�O2��8T���1n��qy4�%}�6���HWg" ����4\\!���C��ھ�D���Cp�֢�kR�[NB���@i\�r���Y��wQc�!� !k̹��_Z#c��AIc<h����ZS��s�DI[w�[�@���4.��H}BxJ��R�񈸺��+���5�N ����jx �G��9\l�Qt�*]��uu����~(p�����1���i�5X\sh:��c�>��G7.�ڌ`v������M�Y�����>�5\���-�R��ǤPV]��w�̾��y��ό���{�K�x)���a�K��C2/����a8v2�6�'B����w�J]Bx�����\��u����眢K�W����*ǽy�LE϶��Υ���� ��R8Fgn����T����XΘ9.�q	��ͮ0��_��a ���4n8��Ś�$_V���?���(�+.()�Q�lqt�ͽ����eK����cӕ�M��%G5�Zv�m�];����X��4Z���;w��%�+b$�R�zSI������͑i��ߣ�6]��2����Ү~���ǋاU�7�ޱ������ǯ�sHNR���ɺȢt4F���}���yT�Qo�L�ڭ�(���+*�`�wU��Jj�K��û���P���h���S�iL5��pm����:�*J��\u�TX�{h����:��5����vB�%E�Il�n��y� ����mT7+��""n�%-��,���o
�W6TM��M�آ��]>Z���>�Q��A���jq]v���u��W���55l��1\Sb}l��
�e�;o�`Ҵ����\�D9n�2,�c�5����ñm��b���hb�N���C��J���:�7s���Է����	�L���W�f���cNQ��P�b�#rfy�Y.���Fۋ��՗��d'�钭_4#��%gh�t�����a��ơ/�8;gg�r����D|�Z3z��.U�V�O4��S���f;�����Wq�fD�_siXQ�s�"���y��&]N���%�8 �:J���P?�� 3Qm�s���s�#�:ׯ[XW#�3<�.�V�PZ����U�������&P̓(��^�?�Qp�R����7B}�x�P�!�GW���
�o"]���Q�Q��0B�S�A�ߵ U-�nJ�P-� H+��ø�:+*��6��T��$2�[��:��p_��{H��@ ��EO�Ez���S�Q���a%�>�(MN0iÕ�N�TKuh��H������D����WP�7�����I���(\���}E�t�}>i}ֽ��/�/��tt�K�ɹ0I�T�Θ��y"7���"B��B$bO$���iT�n�lKucQd5S �]ͽ}ׁns�4K;Iь&3Ǵ+*��S�i�����ڵ]$g�0��a�k⪽��ao��B����M�|!#�ڇo��#G�_����!p��#'�l�������Pƾ���Ŕ��w��"t��n!�7<�c��Z��n�M�*���>3��k��Œ��B��|}"��ލ�F���9�����dBѪ�!�ܷ䓬�n,�"��2AK�ti�~a�-��7�wC�ƌٷ��g��.˝��ߡ1dz�9�I�\�9�#v��߳�j�)�Ձ(ʵ�^̾N�u�wU�TOH�J��9'���J~�O�L�5
|F�eh�\qYf,e������*�ײoP&��Bg�4�ʜ̛YA�<��@%���v�(1��~�X��s3�Θ�9{�p	h��C�jQ}}�5�+�k5����t	J��1���u����鮽�d���&d.i���,�+^w�P�ɥ�)d,����jF}���sǭ�΅L�PO8p��[�K#N�_�Իo��<�A�kl�A�:��������:'�����+nYL��>=�ِB��½9�3�%6b'���N�DX/�{�?��Ɓ�"�%�2����tPO~��GhM�����m�����U���uizE��9�a�F~ ��wu��,��8i'� �F�x�p���vw{����b�%��R��H$ERWq����<oU�syEI��)���w{j�N�ˢT/J����K`�߿\�9�������ka�������׏��7�����+���V�+�Kw._޷�p�r��t��������@+l�<�I�F��b?Q��x���P^eǗ�A�#�K�_w|�we���K����Q?*��񷏾[=�F��܉�yt����]9R��+��U@}x"�'�CG��<�]�~08�:�'��Nv������~�*O@��e�}��-��T�8����aR9��W�;7���u0,:�zڕ��Bu*��,O�]�� -�cׯ!�D}؉���řRϟ���4ׯ��}y���8�e�K���D�
KX�>��Wv�Y#`<Ê���B@=�
�s6_����\(]'�/jQY�e�
�$�5�˭�$n��z}h�;����K���/�*��hi�|-=�S����@�纬��;�9�6��]]8���k�э��@+۶Ѓ+���K=֯�y]ꛮ�y:{ڰ*t9��&���Kf"4xa�~��yC������_�	�&�c��ڍp��x���:��?�b%��U�wm� ����E��佰Zςǎ����յx���u��CG�n�ڃ蟍�׉��:��ݧ�{u�q�,�����ސ�=ȷ����J>�lww�L�C;��z���:/���#���?�+�6�on�Gn��׎�>)�0xd��䐆�J���+M�*�v��u�[W��3�jaO�Wj�\Z��M<���0���т?��N�呎ӼJ���lJ�$C�����f݌�Ew��E�)<Ȼ������G���B���.��!��2̡�I75�h������$�l4��\kؕ>��)wĞ�؇�up��jX��[�:�Qm/<�Zg�^!���BMC��EF����3L��`�r�ړ��������\��Xa��Fˡ�戯���<~�6��yD�_1�����m��m�M��FR1>/͆���7 �����Ȓª�@��<8��\0�,\�Zp��q��io�=�/�?� ߗ<��-ʗ��nD}��-�^I�?Ec7����"�n�&������G�[�X�+@�i��nDe)��s���-��&Ҭsu4������R��ν��xq�߇\��q�**
q�0�(��<����ӮR<���V�g1�Ł��h�9ê�s�������Y=R��[E
�g���g�,�;5�	�=w�䃝�۞^��n-0̭�L}Ϝ���I#j�`��ғP���
D����xD��ޑ Q�(���b(�ۥ���+��s�c����>ba��E�����k�a
���5V�V�0-�]�I����P�VC�Ada���s�A������o�V���ǙҢ����@�G߯p;�	�V������!��ek����#�����2e��nEۻ���<�Q��c%Q[�lu{C'�v�\Z�EBc�]��n/;������ګ&Imq��Gcs�q��+M(��^2���j�q7%|۞�cHj����f�束\/*3P�Zk������r}�N-Q�䆄�Ab��(�Bd���	VP���P�'�<�
�ʭ��GE�=��i"7/>���CsjQOϸ:>�ĝr:ܵ�8�~�M���|����ճ��f�Ά��}hy]�X�������0���iwӳ��r��*��ux֌:��L➷'�o�(>r����y�*���G�o^��".҉���H�.;�2����N��v;�``)���r>��9���y��ޱ;�����B��o���*z�7�o�[���4Q{��l0��Dϔ����xJc�tlf���[�]V6e���	��ي�p�Pz�_��^�Y�:т*����ݰ@�a,�֍�� ^tw�:_\�`�˩����/�=���6�\yW1��'^ZF�ʻ����W���H!��[2�<ݶ&;��.�
s�gg>�z�9�2�� <�L��z�7������p�Fb.FZ]7���M
܍�P�׹�Pg$aOM���o�T�WQ�Y��8�Ŋ���n�,�a�\����J+��wcJ��=��U�Hvh��ܓ0�&�ػ>���0�}%:�\b��q�qn�����k#����5���0>ϱ�K�ș���O�I�X���Ja����X�Z/ݲY�������/s���c��&Y8!|�'�둩��e��4��!��j �9G�X�Tsa�[`�87��a�4ې���]私����u�T�����-�Nl��Пv�%�Vga=�׭��PW�R�1G���Y͎]�OK�<����k���N[��{,'{?E��v���#s�v�C�@������GT{����U3�(E��#=��1����1�6��C�|��RH��&��X��y�/��/�H��DTޔ��5�u&�c�d�5Ti�GHˢ<�߻1U�a�l�QT���ƨt�d"����5�-�j�/��c�_�ݓ��<�	ü~�\�ȭEuJ�WH�<������װ��6FK���6 �#�~����=6�9|�+w�m���� �{
�t�� K|��=-��Y���=Ȁ��0XWwjO���-g�Qa��lCzB��] ��2�VRq��}a�
�����Q��Hϕl��%���yvI�����?��Z�������YE�G���1j��P'��I�3���gCĨ�ZN��%B_LT$�P�ȥy�F�)3MƜ�|Q����7�]e���"Cˠ���Ä.���C725�}�1W�o՟wf��4��ֆ�ĄN�U%�j��P�m��1��wcVx=���Fԁ<��l�v>+�������u�5�z����}z;|���"l�~�m��b���0ӆ!>���7:ln��������Q_�0�^|�8̷f�:���<��5L�-.�`�y`���0�4�`}`Ν�c��8�<?����HܵC����wvH>P�e�ϟZB%�)�2C�ۊ���ϵ��;��Q�R
� Ý�t�h�=B��;=	%.Z���-[�*ǜ���=7��.��È��M�%���k��څ����^9_�<�a2诅]����7T+�uf�Ͷ���u���oA�H�f¤vTi�BB��%q_J�L:Ӯ���N��I��:�{�
��p�����54ã�jh�iTk��~���^t�%ģ�]l���y��HGS�ꢫ�V\P_���4u	¯�-�������*�
$��ؕ������K�֩W|�?�X:�u˒V���-���i?ݵ�(�Aر�0V=���^\v���"(;�{zCu*����Pa�F�b���ۏ�fl��U���ôv�:Ҹ���ݷ�Ѩԑ�E�?Vz
���rD=5\��ڳ���塢�!V}B�{�Gw�+T�><�z���V�Ώ�F��}�ﴻŕ : ��x�ɢ�ځ����+��.O^<�ޏ��E1C$A�Gzϻ�W��v���Oy��z��J��<���ᕉ;S�7iM�{1L�nm��T^Z_^�Q*_�Q^.�Q�0C�Wo�k\y�%�tQ���\Oi|���K=n��A����ZOe��u����
����P�\g��b5�ז$��	��ݶ=�+�"� Vܤ� Tew7�M���[u����|!��+���V`8u;mo�����8xZ��S�:�M��m���P��s�\��"t)��z5�H�n��R�F��J�z#�,���t+��f:]���*W�C T��m����\OU�#b҄��^QOVX�������U#݁h�yq�r�5D������ ��+�d%�{p��ꖻn�.�J^(��I���I>n*s]C����g8��~���F����{�?֮�a��0&:����h����_�,��c�z��;(�Ka(�j�a.��C}�Y1�*?��VJ����}ep����-2��{e�M'8@������^m����݋�2|3���?�>x18������w)��\?�i�}�)٠�QYS�).�-w܊>�s��<�_�^���H\@�}D+����p����F/��R�q w�5�ҥ'�W�FV%�`��H+���8qj|s|Q3/|r�VNn�)��	�5�v�{\�Ճ�_	�	'�:����R��-s�!��U�"w�~�v�C��S����Yu��lg���yeI+��#1��� ���Mc%�q�;��l�ۏ��<q�%0�`�X(<��q��C�k��`Uj��n�G:ƄHӝB;>h�i��� �&�u��-�16�������G��x�
�H�+{�i����n�7�Y�E����?J�~n?����h�t���U ��=m_�K{u��]�  �u�V��a��_>�̍�������!g���;�����;ì}���������X��_�X-\�a�W�p�X�����0~�d���r�y�t�A���'[ʯ>/ܮ�����b�طY�SM���c1�:�U��{Wn�hU�Xy�{hk�T�{�VY�n�*Qw�ywTv�9��U��OϺ��@(�0�G�܆l.��+��\K�Y�-P�B�cmP���<��mP�!`�[�����p�P���(9>���͗6*��|��W�03��\��R�V]�R�V^�����~�2/��t}L�/�����������D�%Q�!k�~ik@��>W�x��-�QQ���!���Y��M��U�Z~�I@�-��ʓ�O���-��;\qas��aU�6�>nQ�<�Q�@�X���|/�c����\���+��W�ۏ)\@�V���T��8��������I�u���6!Ӈ�P��	{��3V~�����ׅ�����ӉO�X�C��-j���J��8Կٍq�X���̩+q�'�b�g��:/w�)�G-��<XhU�ٛPw?8�.��|����V�����+��m���ӓ��@���P�����}e�������0���l+�*�ɵ	��']n��֗�^��o���R�6/\~���M����h���K7]�>�Y�C���#vc�(ݙ��Z��zKcy�� a��b�w��c�@!<��t}���q����uTCg�����['N#Z�J���]�3��'������נu���q3Ә*<r����Z�S��s���V��FwR�م�A=ª/�G4�w���~��o�N�_}��@<'�q<J?v��a0%�������g`c yQ\�6���"*�즾ëd�����Y��a��*ҋ�p��+9K<�ߏ���mDogӇUj{�o�������Ŭ�=Wp_�<�_b��yT9��m�9�s'��	���G�°0�����w�P�Ȝ<_GFG�������+G�p)��Åv��0���[���>�+� �w��kJ�q}��W��;�g�Z� +}�w��|^c9�����){1�UCq'�$�or��_Ӊ��.�c�-T����Ӫ?��JU�r�/���zcm*�-n�30Ǒn�(�=Iw��PE�J2m%#�4te�+�op�U^�h�,.��+��_#�_go������v�
=rh��*�4P/�~�Jr�u3�Z
�� .�rj��[V\WN���<&E�Y�.���銳������{8��(�?��ؑ-��7E�v�w�8�w�?�����C��	N�c���g��-����a�.�ܝ�YG��w��ޕ��#=������;<X-��<�>�����] o�z���%�Op1x9�Hv@ޏ�B(߹Q�Ak)oo/(��F:��}��_�x/61�a��Q�؆4����w_��1��z D��//cD=yȨ�n�Uv`�-odZ��>��.�=��t�jXSt	5X���Ӂ��,n�|���]���U�'dzO��UX�U�|���{3����ח�� �/�8�Y��L�����è�A�|۾|�.BmB���un��|��m\q�N�?�Ù �R��[�j(���2,b�:�2��N��j
��݁uV�pG�㥓�Iw�)�bVe��P!	ٶ]���Wa���b��XnY��W�*`�i���%���n����1�b�ecF�,�Ԋ�XW��H������\CŪc+g�d���yC�<�ns7�~�0�3�����_�O�o������k�_ji�B?(�f�"mű1�~�~-�s;�l8��ݴ{����Rpj_�ٿ��xrLʯ�T�ۂe֮��L�e�"�]��0���X��r]PUI�Gl�}�V�y/z"��&���c�����&PC�ו��rM7����a4p8�"Կ^�W�T�'*�^��ZZ_�y�����AT������o�}��2��n���0/o���+�tM������r�P���&�?3�῾�weF��wy�9���eU��y����¿�ܫ~�2x������a�|}]�w�� �FJ�+���1T��ƿ&�l�U�ƴ�U�AѦ�7�Wn�,f�X���f�F����?
�ۿ �!W�bk/��߆q�֦����2�o�\�~oo@���$m-Tz����mS���߬�`bGp�;8v�,��I�}�]O��a�	C���n/�ά��M���p�W����K�(�����^���U~�PQ<���3� ?%jXϪ�8��#��=B�=q�ׄ��|�a�)~Qp�R~:�7�xS��%i}S'�A��[	�_Eԯ�@����A����j���c�k�[u-n�;E�7���;ˋ�����G�K�n�n����6����� �`U~D�m|�z����~����a�ǝ���N���H��B+������<��;[`�c��2�������@���*,���'�~���>q������(�c4���
�0
�-7���n;�8?zp���S}}�m2�ʖ��Ԕ;����;�z�s3��sa���Yw;�v(iB�O[�0��2I��:=�z~ԙ�wޟ��1�fm8��k&/�\�I�(VW/�;��t�_�3�ɦ�!�pW������� r|U���u���i��;�&%
�V��܉/$�y�І�4��/�b~�O�p�x��jKC,}j�FȑC��v%+Aܛ�W�C:'���Ot!�7޲@8LWk2*��I�n+��2�p��VG}TGM�Qc��d	5�ΦSU��S���B��9�ߎ��[�2�G��0|�lC��8�~���n�B��O����#)�B�����[ŷѐ�0�¢��\|Gl:��L���s8���k�Lc�ؔ;��@t�w�������0�9��Fr:�N�h�;:�#*w._��{����G�&N�5��]��q�D@ń�L��r��1���o�L@�@�J�n�����ɽ���a���C������Ig���s�6���΋����k��c���Sk��K�I�9�*z�+G��E[8ӕsPTp��\0"�J������Ј��&J�wQL w����~x	4+S#m��pdB�m��S�.�T �1�x�Â��¸8m�����)x<(�Q�;��`���%�R^��F��Q�.A�8�ll�+6P��:[��n�׶01�1��W�ԭ�`��!�����n-X�
�۬r��r/O����b�?��~����3x���
7�����k���z���	T���8#���$���qw���B�
_|&�NUm<<����V,���{���@y�i��x�ã,{���v77 b_��`�6����wG�#��*&L�t~�����e�Ǡ�gv�`�iB��ċuh���œ}�/��������5.� �F�<v/�	�s�a����|�]{¾f|�"�.9{P�O��'�աP��G�XZsH�� UO4�����mb��E��+z[;SU\{%Q����yJb 9��w`�����/����T5Q;���d����h ��J�tK?��Oh��uܵ��O!=��%���E�|O�l��) U`���쉉�5�]���zt�:��=����aQX?��ke	���� (�{�ͽ�~�鿣���>���e|�������b�1"�Ag��/��7���N�t+����.h����
S4w
�Sv���N1�b��Z8��N �)-���D0{�Ng�������&r�K������>�gӕ7Q%�=⁙c��$�V���ɡA[����?��^��:�p����k����Ns�OOEF�͸W�<_�Pg���F�RZ_�xu^JV�U��D�"{{Ğk���ه=���%�i�A�E.��Yw�?�rz/���p���,������������?�+�w����q�(M3�{C�fo��*���/�"��ǰ�֝��ڊ����n�F����͔��sB�::���}q'R���ag��)�� [����fz���]���� k:���ӓ�'�J�����뛘^>�l�+��m���:�.[1��߶!綜w�WQ���-���0����z�
=��#�-�n���ʫ���mo����n�ƭ\G�ݺ�b��/ː�ay\��FP�}�!��r�*ʫ�����`=E�C���/�X'��=VC��� m`��u��J���!w���7��rV�tH��@�X��="�?<�-`�пb�?�1�Al���mcL�;=���_�a��ygؽ����W�Nk+|�w&�ɿ�����U�X��d:�5n/�Ґ��ߵ�!��!9}ij�m��nnH`�b�W���n4�/�)��'Yz��ܙ�}ð�r݋�,s�Q�����n�Q�W7�$���0�ڄ�K��ꂔ�^{p?�[�C��Ч	�VR���W��3�R�����Q�Ǐ����)qo�U�Q�Zݥ�B��>���r`D���X�TTCDl���U��B����So�A	�؁ވ����+o�����FGy��S���'\|��k��׋7��X<�� *�Z���/����k�70t�~�����n�!���M���z�t���3��������/�/U��+T�?�O��)�W���M7B����R�j N��x�=^�7�x�)07����#�%��tG�7P�ܳ�|�����}�ck>������a�
�]�%G��9��C/�����\][`��ݽ��;\�Dԅ�a�>r����z��J�|�l���Y�/��ܚծܳ&4|�u���Z����<j�/�ln]�Ba���m���9t�*������<�ʰ�9w�a>v���1��@��r�˰��pPY���'��Q��$ü�h�i5��"���u/�V��Z��p��n�J�S�
N�����{ݼ�n�Q��ğ��4kFhթ�Ϫ<׵J��r�+>Cs�u��Hf-\�_�v�����ϔ	��U�p97'�V��5X
?��Y��);�p|��-o�\J�K.�H?��	�ie��_�����,.8V�˓:���=��OC�^B��ݻ�C(�r����7x���1qpzl�E���PRx�t��܉�Zi�_x��{�۽qSw�B�2�Ki��BO3js�����@�7&pBt��
��;�)���
Zui�qaP:�W���a���������ps�1�Wu��B�:����B����<v�&9aV���}���N�遫8O����$�����Ƒ���2S,��qs�r�Xn�Yb/'0>�"�[zݖ]��y���	�'�U�Nqj^��q���ܖ/�Ǣ���W\w�Lӎ^ۍ��D����^��9��x��	�\_�־�
!����-��M�a`\�<r���4�'\ϵ��!�B�Ft����/ۻ�3�����ڽ���	4%��=���vg��'�#�MR��%OVU!�R�k�ER١'ᛣ�X��>��k�L,	C����,O�@�"j{;|d�7��MԾ���w�!��8�WU^��'�Z�!ԜVo�:*��/�K'�2@�,�bET�P� 1�i�g9,_B୔"+�oPQ/�ZL��r�a 3W�3q�ZQb�M5����6E�@2��+[���*$T]�"S���'�f����*ֲii�:�DMPSZU�<�q��W�KSW��ď�:��N?� {��:�����B~2j�QS���o-��������|A�lQ=��.-��v{��^~���W^|U�hxS��'1�QC�m�M���pbm���+��h�>�O�j�2*��G��C9��5�~9����\�ʻ�M9T�P,������$2��g��
��Պ��+���������՛���g/UO�}>TZ��5|I'f�r�^�ρ
�v�]NkU>�΃W��!�r},W�B�5մ����+�ЕbRC���FT�.E5-߀
i�#�@�����jR�QIFT�PkJ)(b�L�}Ts�c]m�a�ޘMQY8��C�,cK_W��G��:�IT��X��_L��*+�D?���Qi�t��7A�z5]��5�5ռxc+�
�/[�RP�\cx2j�VQSK�!���i�YE)��v�?eME�Bše�zCW��d|��h�Pi{Yї�0�z�g!i�b_S�Qu��qYԬ�/��"�$yE �����k�jQ�!*u�)�i����gE����KQ۟�<�^:5��-:��$�ҳ�\f14�M�I�U%�,�%�5Z	�ȿ ��RT�4�Ruy�%u��uI{m�jp�P�@9ݘÍ��q|R�J�<�T���[�z`��)W�(@d�+��"hoϨ&O1{��lƾ�ۀ���^�P��K���3MKPIP�	�7�$��Ɨt�-1�`'��rϨ��ꓦ(���c�/H#*b�TTg���0]�jZ�)��VQ�&����̨�1�H��F)x�6�����6�� Ws�B� 2oZ�������Q��gD��J{���>�?!�c,�lu�v%]r_Ae���̷}�V�4�f��+T|\�M�%3��<XU_��
ۘ/KP�BWD��\�fi�)����"��=b΄���g�7�!�Ө�*�l�����b_qhNt���C��uT_EEX�QR`d��D	!|�hDUG��Qe�@�9�����h�(���"�.aH�>b  w���29�(�����Q��K�����J�%H�I�_A%�Q��Qe�Q�_�aL�PI뉨��Ul��}5i�o��o�W�i�Vj�01�)�IUJ�,|�7�%������	 I��,IdfUٸZ^g{/�Y˩84B��i`� ��H��<}K��&i���� RI6�X!s2/U+�V��K#��)j�8*�Ϸ>�qE35��)7���p����|ю�a�P��@R�ުX*|=j����
=Y3��|����4�j$���gN;[�L��ҫ�2. "j�i�����w	S�5�U�9�fԪ�ݧ�����
�t�S��I�YQ�z���H6���˪�*s���̡'�j���v{�}�_Q�^��-E��c# am��eV
�7�<#���<S@T(�����
,+��
�gFm�Vzm�
'p	��M5��*UT�2(C�-#��Qab��)Q��qK�R6gB��I*����'�K�*���Ԗm5 xŗPA����:L�L)�v��nyQ�9�2��Ѐ*�\S��Q�1u����i5',�(�}E)Q*�h�M}����M�']V�b���س���R��D���U�������f�݀�/� ���׊��Z2��[����Yǔ"#y���Ӏ���[�ȈJ��}B�*�=e���5�he�#2yL��*��OCU��eP�=YTx^>���T3l@5ߦ�/�j�q�Yh�j��ī�Wi��QC�k����3&Ob�R��/�2/��j��Ϩ��OG���ς
��Pc�Ј�zZ�)�5Ae�&�.�UQ!ˬ��rxiZ���<Y��4�PV�q�4�F�g����!��7D�+Y�!�(L�^��������4*	*�FM���H��YQ���W��q$}B#��m	����W�Ϫ5TP�TQ������9
�@s��;����CZ��42j󵖓�k�,��5�STo��jD���TP��䰂
�o�����UN͋QB%i�&���� I�b�8�xgT�VQ�KQ�N�dI=�\�oU;�z�I5Kk�V� _�'K����Լ�E�-*�Z���c��Q�G��U��[{���rOX�a���Q@�	��zUH����t�9o9E#�ZЉ)��A쭁J��3 8�Q�CO�2��2��l��P�9�5��2��<;*���-��!0'�sEM�5�[�-M+���FS�t�_>p,���
�Q�rQjց������?��"ZE�\�a	j�r�r	��Q���6�xg% ��VQKPc
h%ˣ��T�︩�f��'��FԊ$�U3�=9b<������ IB`ʪ(pc�,��VHĴJf>���*R�H��L���5y�29�BL�DX�7C5������	�v�,���V��Hf,���kv�T~��=�.�GuT�B�<�_�<YI3�E�ڒHdQ;�fT13�SQs��I�YQD���W���f�r9�f�[o"C�ူڤX�-�Vb�z�ɨlz�I��QǛvBE@�:j�[�j���砚�gCe�&����m��Iڼ\#���
	�J��M�����6IHԊ��S��)UU#sz�)'�Ϩ�GԌ@��%�,c��	�+	��jK�
ˡ���ZZst�?�Pup1v�uT�LQy���$Q)W)���g��'�֢2���+�N:�y��,��JQ�#��0|&Td05qE�Z3���հ��7C��\GB ?�
-�u)j�Z%<52����(�$�Q�PI����N6p|F�)4Td�d)�QqS��QTQ�DG�� r="�#�uk�z�L��{X�[Q�2�ZM�#[��P�M�G%˝��e���dԸd���J�k�E!d���2E�xzbOUGTȡн<ďs��	/z	���
�$����DZ�"Xh�JWˠz�"+�Ce��֜�Q�/��O@�?OF%S���4�d���%���,XQy��J-P�f%/UiƸ��ޞ���u1YFi h�j�KPU�PG�r���D��i��VP��&uXK�AW��s�M�%զ�Q�Vz"R[�
,depQ��Ts8�7�ҡ��XETS�����Q_�V�|�R Xr�j2_WJ�9��+���Oo�u����@�ˡz����]����F��>/��>�fwj���#��bEW*(/T�S쥔���8	�U��o�B�?��QI���n�$�R]�U��{**��(�J�{)��%9/߾�ùG�Z@2	���OY-��."�\C@TE8��.%�P5�%��3و��eYd/�
*����4TrԊ�$-��m��*���T��5��(�*Ԥ�&�9�`�#-��Iu�(���t�<D�Ѣ�1&}��-�* F=��j��ZƝ��T#	��H<��UT���f(T��?�������Q+�RހJWˡ"��R��)�a-[C�� m�v'J+*�7EJ$�Zp����aZ�6�%��
�W�>�*�j!����U��� �,��0�a*�̡91�h��B�YKaY��Ī�fd�#�d�R,�emҿ���G*%���x�mD�d���G�&�&Oy���H`�']�_�jƐ���Pë�� 2�m\��� j-x+2��KDCcȉ����숊P�G�R�I˩���Ҫ�J�	���`X��H�dsT����A�}S�l�PCui��̨���KCZsMZAe��b2�������UoPY��J&Kղ���WC��vӵ:�*v�6�� �/4#�rGr������,�b��ᤊ
��C��ZE�	*,�Jm]�Ur�"��C,T�bmBt��1Eմ"V��wu>BN+���I0TXɯ\3^6���P-ݰ˨R����hH$)S�;J�eqQB xj�RT+C�"���Q���3����*s��}VTa�2|Ib��6y�4|��[��P�^�hf�d���`r�rm"�P�
*�T�4TfDJkB'�K�πjiE)DTI��>�0=	��<��iAT�a�E�H����P���o�	�ɁxJ�H�"Ոj�B	� ���iy�����431I�V���^�V!J�6����M+��w�!���5TeTT^$\��OL+ش�=�πj���k�,l�ȏ�k{B��&:��P�6�Q�#�%�j��XK+�T�
YTQ-}5T2Q�镨 �1B��Qu�;YD��gb�[Ṯ�UԢ�:E�Z�k�j�9H��䲗�f��������UHo�xo	2_KPs�deT~gT��QB�!N<r��P#wσ�A����H"�Ƣh�H�i�ٽ�3j����U}@�Z���DH�؂i�؅$��\��B��������L��DB��1TpqK��j�ɨ�'�	oN����U�tȻ:�o��J���<�5�q�*�(I#���SQ�,���TQ#��Q}����Q#W��Q��_ 5�}.���ʤ�W+�ץ"���˓�+M����SjC�:l�ڴ��P�,��dE�d�d�\��U�
����K5�C݄�ǄJ��q�=��:j�HY�ᇾ�V�7�dI�����#kH_ဠk
|�6��r�&�>����eQ�#��Z����IT�%!�U���2�*�V���5�$�S��iB��o��k]�
ad�&j��+j`IC���.協���(Z	�1��ڔ�9*줍�&��m��r!#�ʊ:Q�I�@��"��h�[N*���Z��[^�I3O�dI�A�RFa�cԏy��q*�ЋJ�F�ĨF8yL�4��
jVZ�|�P��-5�>j�'˘ޘKQ�Q�������*�u�j|"]%T�Q�y+TW��i%Y�G7{:	�=�5MӉ�1�ы7����
�7qԷ(e�s�RK�:*M��ɨ��ԘISZ����kuBD]9X����kU�|DF#"���o�DR�T��! �����T3|:*Q���v��H欉�UTo�'�.�V}j��c^�ꛣ� �GT_C���j;��M�Q�P�,��D��88��>k$�m��	B?ri���$�T��;:�#d�d<+*���/C������d�/�
�IDT��[U�����N��	BKِ�\9e�*cF�Z&� E	J^꺓�Tπ&�I��QI2�� O9#%�,?2$T|%T�>jLUҮ����\�ha͉�ZvWUrVe7�%�2E
@EMpFP������`�{o���t�<��K�D-ݒSEe=kԥC��i��TD�h(�FO����Q�\#j�1TQ�Ö8�Q�k3T�ao��̽'�r8[Zr+o��y������D+��T�J)4GU	��L�uEqu�/�3����ݎ��OCHK    "�
     X       P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��     �   1��VOCHK            �H    
   is_result              @                                P       DIMENSION_LIST                                    �     6   규=       ��            (�
            b�A)OCHK            �4       _Netcdf4Dimid                          tL$SOCHK            �4       _Netcdf4Dimid                          tL$S         ��            ���FHDB jk        �,OT       systemwide_capacity_factor��     U       systemwide_levelised_cost(�
     V       total_levelised_cost��
     �       timestep_resolutionE�     �       timestep_weights��     �       resource_unitG�     �       energy_cap_minv     �       
energy_eff��     �       energy_ramping��     �       storage_cap_max|     �       
energy_con�(     �       lifetimen     �       energy_prod�+     �       energy_cap_maxq*               TREE  �����������������                               ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OHDR�8                                                       ?      @ 4 4�              �            �8             shuffle            deflate            ��
                      ������������������������T        _Netcdf4Coordinates                                               P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               p       DIMENSION_LIST                                    ��     �      ��     �      ��     �   4       _Netcdf4Dimid                          zr��OCHK    &     `      4       _Netcdf4Dimid                          ɠ�[OCHK    g     �      4       _Netcdf4Dimid                          "5TREE  �����������������                                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         OHDRM(                                         ?      @ 4 4�      z�
     X                   �8             shuffle            deflate            .�
                   ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                               `       DIMENSION_LIST                                    ��     �      ��        VP�dOCHK    �            �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ��             (�
             ��
             ��rTREE  ����������������                               P�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          )   &�
     \   )   &�
     [   )   &�
     Z      &�
     X      &�
     Y   )   &�
     S   )   &�
     T   )   &�
     U      &�
     V      &�
     W   $   &�
     J      &�
     K   %   &�
     L   %   &�
     M      &�
     N      &�
     O   $   &�
     P   )   &�
     Q      &�
     R   $   &�
     7      &�
     8   %   &�
     9   %   &�
     :   )   &�
     ;   )   &�
     <   )   &�
     =   )   &�
     >      &�
     ?   $   &�
     @   $   &�
     A   )   &�
     B   $   &�
     C   %   &�
     D   )   &�
     E   $   &�
     F   )   &�
     G   )   &�
     H   )   &�
     I   )   &�
        %   &�
        %   &�
        )   &�
           &�
        $   &�
        )   &�
           &�
        )   &�
        %   &�
        $   &�
        )   &�
           &�
        $   &�
           &�
           &�
            &�
     !   %   &�
     "      &�
     #   )   &�
     $   %   &�
     %      &�
     &      &�
     '   %   &�
     (   $   &�
     )      &�
     *   )   &�
     +   )   &�
     ,   )   &�
     -   )   &�
     .      &�
     /      &�
     0   )   &�
     1   $   &�
     2   )   &�
     3      &�
     4   %   &�
     5      &�
     6   GCOL                                                                                                                                   	               
                                                                                                                )       NL41::hvac_transmission:NL34::electricity              %       NL21::demand_electricity::electricity          %       NL22::demand_electricity::electricity          )       NL21::hvac_transmission:NL22::electricity                     NL23::curtailment::electricity         $       NL41::hydrogen_to_power::electricity           )       NL32::hvac_transmission:NL33::electricity                     NL13::curtailment::electricity         )       NL23::hvac_transmission:NL21::electricity              %       NL12::demand_electricity::electricity          $       NL23::hydrogen_to_power::electricity           )       NL22::hvac_transmission:NL21::electricity                     NL31::battery::electricity             $       NL22::hydrogen_to_power::electricity                  NL31::curtailment::electricity                 NL41::battery::electricity      !              NL23::battery::electricity      "       %       NL13::demand_electricity::electricity   #              NL12::battery::electricity      $       )       NL33::hvac_transmission:NL41::electricity       %       %       NL23::demand_electricity::electricity   &              NL34::curtailment::electricity  '              NL41::curtailment::electricity  (       %       NL31::demand_electricity::electricity   )       $       NL11::hydrogen_to_power::electricity    *              NL12::curtailment::electricity  +       )       NL31::hvac_transmission:NL32::electricity       ,       )       NL12::hvac_transmission:NL23::electricity       -       )       NL22::hvac_transmission:NL41::electricity       .       )       NL21::hvac_transmission:NL13::electricity       /              NL32::curtailment::electricity  0              NL11::battery::electricity      1       )       NL23::hvac_transmission:NL12::electricity       2       $       NL12::hydrogen_to_power::electricity    3       )       NL13::hvac_transmission:NL21::electricity       4              NL42::battery::electricity      5       %       NL41::demand_electricity::electricity   6              NL22::curtailment::electricity  7       $       NL21::hydrogen_to_power::electricity    8              NL13::battery::electricity      9       %       NL32::demand_electricity::electricity   :       %       NL42::demand_electricity::electricity   ;       )       NL21::hvac_transmission:NL23::electricity       <       )       NL41::hvac_transmission:NL33::electricity       =       )       NL33::hvac_transmission:NL31::electricity       >       )       NL11::hvac_transmission:NL13::electricity       ?              NL32::battery::electricity      @       $       NL31::hydrogen_to_power::electricity    A       $       NL13::hydrogen_to_power::electricity    B       )       NL34::hvac_transmission:NL41::electricity       C       $       NL33::hydrogen_to_power::electricity    D       %       NL33::demand_electricity::electricity   E       )       NL42::hvac_transmission:NL41::electricity       F       $       NL32::hydrogen_to_power::electricity    G       )       NL11::hvac_transmission:NL12::electricity       H       )       NL12::hvac_transmission:NL11::electricity       I       )       NL33::hvac_transmission:NL32::electricity       J       $       NL42::hydrogen_to_power::electricity    K              NL22::battery::electricity      L       %       NL34::demand_electricity::electricity   M       %       NL11::demand_electricity::electricity   N              NL21::curtailment::electricity  O              NL42::curtailment::electricity  P       $       NL34::hydrogen_to_power::electricity    Q       )       NL31::hvac_transmission:NL33::electricity       R              NL34::battery::electricity      S       )       NL32::hvac_transmission:NL23::electricity       T       )       NL23::hvac_transmission:NL32::electricity       U       )       NL13::hvac_transmission:NL11::electricity       V              NL33::curtailment::electricity  W              NL11::curtailment::electricity  X              NL33::battery::electricity      Y              NL21::battery::electricity      Z       )       NL32::hvac_transmission:NL31::electricity       [       )       NL41::hvac_transmission:NL22::electricity       \       )       NL41::hvac_transmission:NL42::electricity       ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               NL11::wind_offshore::electricity�       $       NL31::hydrogen_to_power::electricity    �              NL41::lost_load::electricity    �              NL23::pv_farm::electricity      �              NL11::lost_load::electricity    �       $       NL33::hydrogen_to_power::electricity    �              NL31::pv_rooftop::electricity   �       )       NL33::hvac_transmission:NL32::electricity       �       )       NL32::hvac_transmission:NL33::electricity       �       $       NL22::hydrogen_to_power::electricity    �              NL22::lost_load::electricity    �              NL32::lost_load::electricity    �       )       NL33::hvac_transmission:NL41::electricity       �              NL22::wind_onshore::electricity �              NL41::pv_rooftop::electricity   �              NL33::wind_onshore::electricity �              NL12::pv_rooftop::electricity   �              NL33::pv_rooftop::electricity   �              NL11::pv_farm::electricity      �              NL23::pv_rooftop::electricity   �              NL23::ccgt::electricity �              NL11::battery::electricity      �              NL32::pv_farm::electricity      �       )       NL41::hvac_transmission:NL22::electricity       �              NL34::wind_onshore::electricity �       )       NL32::hvac_transmission:NL23::electricity       �       )       NL23::hvac_transmission:NL32::electricity       �              NL21::ccgt::electricity �              NL13::ccgt::electricity �       )       NL31::hvac_transmission:NL33::electricity       �              NL34::battery::electricity      �               NL23::wind_offshore::electricity�              NL34::lost_load::electricity    �               �                  �     U      �     T   $   �     R      �     S      �     N   $   �     O   )   �     P      �     Q      �     G   )   �     H      �     I      �     J      �     K      �     L       �     M   )   �     8   )   �     9   )   �     :      �     ;   $   �     <   $   �     =   )   �     >   )   �     ?       �     @   )   �     A   )   �     B      �     C      �     D      �     E      �     F   $   �           �           �           �        )   �        )   �             �     !   )   �     "   $   �     #      �     $      �     %      �     &      �     '      �     (      �     )       �     *      �     +      �     ,      �     -      �     .      �     /      �     0      �     1      �     2      �     3      �     4   )   �     5   )   �     6      �     7       &�
     �   $   &�
     �      &�
     �      &�
     �      &�
     �   $   &�
     �      &�
     �   )   &�
     �   )   &�
     �   $   &�
     �      &�
     �      &�
     �   )   &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �      &�
     �   )   &�
     �      &�
     �   )   &�
     �   )   &�
     �      &�
     �      &�
     �   )   &�
     �      &�
     �       &�
     �      &�
     �   )   �           �           �        $   �        )   �        )   �           �        )   �           �     	      �     
      �           �        $   �        )   �           �           �           �           �           �           �        $   �           �           �        )   �        )   �           �        GCOL                  )       NL21::hvac_transmission:NL23::electricity                     NL11::ccgt::electricity               NL31::pv_farm::electricity             $       NL13::hydrogen_to_power::electricity           )       NL34::hvac_transmission:NL41::electricity              )       NL42::hvac_transmission:NL41::electricity                     NL42::lost_load::electricity           )       NL12::hvac_transmission:NL11::electricity       	              NL13::wind_onshore::electricity 
              NL34::ccgt::electricity               NL12::lost_load::electricity                  NL13::battery::electricity             $       NL23::hydrogen_to_power::electricity           )       NL23::hvac_transmission:NL21::electricity                     NL31::lost_load::electricity                  NL21::pv_rooftop::electricity                 NL13::pv_farm::electricity                    NL23::wind_onshore::electricity               NL32::pv_rooftop::electricity                 NL32::wind_onshore::electricity        $       NL11::hydrogen_to_power::electricity                  NL33::pv_farm::electricity                    NL42::pv_farm::electricity             )       NL22::hvac_transmission:NL41::electricity              )       NL23::hvac_transmission:NL12::electricity                     NL42::battery::electricity             $       NL34::hydrogen_to_power::electricity                  NL21::lost_load::electricity                  NL32::battery::electricity                    NL34::pv_farm::electricity             )       NL11::hvac_transmission:NL12::electricity               )       NL41::hvac_transmission:NL34::electricity       !               NL33::wind_offshore::electricity"       )       NL21::hvac_transmission:NL22::electricity       #       $       NL41::hydrogen_to_power::electricity    $              NL13::lost_load::electricity    %              NL33::lost_load::electricity    &              NL42::ccgt::electricity '              NL12::wind_onshore::electricity (              NL21::pv_farm::electricity      )              NL41::ccgt::electricity *               NL12::wind_offshore::electricity+              NL11::wind_onshore::electricity ,              NL31::battery::electricity      -              NL11::pv_rooftop::electricity   .              NL23::lost_load::electricity    /              NL31::ccgt::electricity 0              NL41::pv_farm::electricity      1              NL42::wind_onshore::electricity 2              NL33::ccgt::electricity 3              NL31::wind_onshore::electricity 4              NL22::ccgt::electricity 5       )       NL33::hvac_transmission:NL31::electricity       6       )       NL11::hvac_transmission:NL13::electricity       7              NL22::pv_rooftop::electricity   8       )       NL31::hvac_transmission:NL32::electricity       9       )       NL12::hvac_transmission:NL23::electricity       :       )       NL21::hvac_transmission:NL13::electricity       ;              NL12::pv_farm::electricity      <       $       NL12::hydrogen_to_power::electricity    =       $       NL32::hydrogen_to_power::electricity    >       )       NL13::hvac_transmission:NL11::electricity       ?       )       NL13::hvac_transmission:NL21::electricity       @               NL34::wind_offshore::electricityA       )       NL41::hvac_transmission:NL42::electricity       B       )       NL32::hvac_transmission:NL31::electricity       C              NL34::pv_rooftop::electricity   D              NL21::battery::electricity      E              NL41::wind_onshore::electricity F              NL12::ccgt::electricity G              NL13::pv_rooftop::electricity   H       )       NL22::hvac_transmission:NL21::electricity       I              NL21::wind_onshore::electricity J              NL41::battery::electricity      K              NL23::battery::electricity      L              NL12::battery::electricity      M               NL32::wind_offshore::electricityN              NL42::pv_rooftop::electricity   O       $       NL21::hydrogen_to_power::electricity    P       )       NL41::hvac_transmission:NL33::electricity       Q              NL32::ccgt::electricity R       $       NL42::hydrogen_to_power::electricity    S              NL22::battery::electricity      T              NL22::pv_farm::electricity      U              NL33::battery::electricity      V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o              NL13::curtailment::electricity  p       %       NL12::demand_electricity::electricity   q              NL31::curtailment::electricity  r       %       NL13::demand_electricity::electricity   s       %       NL23::demand_electricity::electricity   t              NL34::curtailment::electricity  u              NL41::curtailment::electricity  v       %       NL31::demand_electricity::electricity   w              NL12::curtailment::electricity  x              NL32::curtailment::electricity  y       %       NL41::demand_electricity::electricity   z              NL22::curtailment::electricity  {       %       NL42::demand_electricity::electricity   |       %       NL33::demand_electricity::electricity   }       %       NL21::demand_electricity::electricity   ~              NL33::curtailment::electricity         %       NL22::demand_electricity::electricity   �              NL23::curtailment::electricity  �              NL21::curtailment::electricity  �              NL42::curtailment::electricity  �       %       NL32::demand_electricity::electricity   �       %       NL11::demand_electricity::electricity   �       %       NL34::demand_electricity::electricity   �              NL11::curtailment::electricity  �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL22::ccgt::electricity �              NL23::ccgt::electricity �              NL21::ccgt::electricity �              NL13::ccgt::electricity �              NL11::ccgt::electricity �              NL34::ccgt::electricity �              NL41::ccgt::electricity �              NL31::ccgt::electricity �              NL33::ccgt::electricity �              NL42::ccgt::electricity �              NL12::ccgt::electricity �              NL32::ccgt::electricity �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL21::pv_farm::electricity      �              NL34::ccgt::electricity �              NL31::lost_load::electricity    �              NL21::pv_rooftop::electricity   �              NL21::wind_onshore::electricity �              NL13::lost_load::electricity    �              NL23::wind_onshore::electricity �                          �     �   %   �     �   %   �     �      �     �      �     �   %   �     �   %   �     {   %   �     |   %   �     }      �     ~   %   �           �     �      �     o   %   �     p      �     q   %   �     r   %   �     s      �     t      �     u   %   �     v      �     w      �     x   %   �     y      �     z      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �B     ;      �B     :      �B     9      �B     7      �B     8      �B     3      �B     4      �B     5      �B     6      �B     +      �B     ,       �B     -      �B     .      �B     /      �B     0      �B     1      �B     2      �B           �B           �B           �B            �B           �B            �B     !      �B     "      �B     #      �B     $      �B     %      �B     &      �B     '       �B     (      �B     )      �B     *      �     �      �     �      �     �      �     �      �     �      �     �      �     �      �B           �B           �B           �B           �B            �B           �B           �B           �B     	      �B     
      �B           �B           �B           �B           �B           �B           �B            �B           �B           �B           �B           �B           �B           �B           �B            �B        GCOL                        NL13::pv_farm::electricity                    NL32::pv_rooftop::electricity                 NL32::wind_onshore::electricity               NL22::lost_load::electricity                  NL32::lost_load::electricity                   NL32::wind_offshore::electricity              NL22::wind_onshore::electricity               NL41::pv_rooftop::electricity   	              NL33::wind_onshore::electricity 
              NL33::lost_load::electricity                  NL42::ccgt::electricity               NL12::pv_rooftop::electricity                 NL33::pv_farm::electricity                    NL12::wind_onshore::electricity               NL42::pv_farm::electricity                    NL41::ccgt::electricity               NL12::pv_farm::electricity                     NL12::wind_offshore::electricity              NL23::pv_rooftop::electricity                 NL23::ccgt::electricity               NL11::pv_farm::electricity                    NL33::pv_rooftop::electricity                 NL11::wind_onshore::electricity               NL11::pv_rooftop::electricity                 NL23::lost_load::electricity                   NL34::wind_offshore::electricity              NL22::ccgt::electricity               NL11::ccgt::electricity               NL31::pv_farm::electricity                    NL22::pv_rooftop::electricity                  NL11::wind_offshore::electricity               NL41::lost_load::electricity    !              NL23::pv_farm::electricity      "              NL11::lost_load::electricity    #              NL31::pv_rooftop::electricity   $              NL32::ccgt::electricity %              NL34::pv_farm::electricity      &              NL42::lost_load::electricity    '              NL13::wind_onshore::electricity (               NL33::wind_offshore::electricity)              NL12::lost_load::electricity    *              NL13::pv_rooftop::electricity   +              NL41::pv_farm::electricity      ,              NL12::ccgt::electricity -               NL23::wind_offshore::electricity.              NL42::wind_onshore::electricity /              NL33::ccgt::electricity 0              NL21::lost_load::electricity    1              NL34::lost_load::electricity    2              NL31::wind_onshore::electricity 3              NL21::ccgt::electricity 4              NL13::ccgt::electricity 5              NL31::ccgt::electricity 6              NL42::pv_rooftop::electricity   7              NL41::wind_onshore::electricity 8              NL22::pv_farm::electricity      9              NL34::pv_rooftop::electricity   :              NL34::wind_onshore::electricity ;              NL32::pv_farm::electricity      <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                             NL41::wind_onshore::electricity �              NL21::ccgt::electricity �              NL13::ccgt::electricity �              NL12::ccgt::electricity �               NL23::wind_offshore::electricity�              NL21::lost_load::electricity       W     <      W     ;      W     :      W     8      W     9      W     4      W     5      W     6      W     7       W     ,      W     -      W     .      W     /      W     0      W     1      W     2      W     3      W           W            W           W           W            W     !      W     "      W     #      W     $      W     %      W     &      W     '       W     (      W     )      W     *      W     +      �B           �B     �      �B     �      �B     �       �B     �      �B     �      W           W           W           W           W           W            W           W           W     	      W     
      W           W           W           W           W           W           W           W           W           W           W           W           W            W           W           W           W        GCOL                        NL34::lost_load::electricity                  NL11::ccgt::electricity               NL31::pv_farm::electricity                    NL34::pv_farm::electricity                    NL42::lost_load::electricity                  NL13::wind_onshore::electricity                NL33::wind_offshore::electricity              NL12::lost_load::electricity    	              NL34::ccgt::electricity 
              NL31::lost_load::electricity                  NL21::pv_rooftop::electricity                 NL13::lost_load::electricity                  NL23::wind_onshore::electricity               NL13::pv_farm::electricity                    NL32::pv_rooftop::electricity                 NL32::wind_onshore::electricity               NL33::lost_load::electricity                  NL42::ccgt::electricity               NL33::pv_farm::electricity                    NL42::pv_farm::electricity                    NL21::pv_farm::electricity                    NL12::wind_onshore::electricity               NL41::ccgt::electricity                NL12::wind_offshore::electricity              NL11::wind_onshore::electricity               NL11::pv_rooftop::electricity                 NL23::lost_load::electricity                  NL22::lost_load::electricity                  NL32::lost_load::electricity                   NL32::wind_offshore::electricity              NL22::wind_onshore::electricity                NL41::pv_rooftop::electricity   !              NL33::wind_onshore::electricity "              NL12::pv_rooftop::electricity   #              NL33::pv_rooftop::electricity   $              NL12::pv_farm::electricity      %              NL11::pv_farm::electricity      &              NL23::pv_rooftop::electricity   '              NL23::ccgt::electricity (               NL34::wind_offshore::electricity)              NL32::pv_farm::electricity      *              NL34::wind_onshore::electricity +              NL34::pv_rooftop::electricity   ,               NL11::wind_offshore::electricity-              NL41::lost_load::electricity    .              NL23::pv_farm::electricity      /              NL11::lost_load::electricity    0              NL31::pv_rooftop::electricity   1              NL32::ccgt::electricity 2              NL13::pv_rooftop::electricity   3              NL21::wind_onshore::electricity 4              NL33::ccgt::electricity 5              NL31::wind_onshore::electricity 6              NL22::ccgt::electricity 7              NL22::pv_rooftop::electricity   8              NL41::pv_farm::electricity      9              NL42::wind_onshore::electricity :              NL42::pv_rooftop::electricity   ;              NL31::ccgt::electricity <              NL22::pv_farm::electricity      =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V              NL22::battery   W              NL34::hydrogen_to_power X              NL21::hydrogen_to_power Y              NL11::battery   Z              NL33::hydrogen_to_power [              NL41::battery   \              NL13::hydrogen_to_power ]              NL42::battery   ^              NL31::battery   _              NL22::hydrogen_to_power `              NL41::hydrogen_to_power a              NL13::battery   b              NL34::battery   c              NL12::hydrogen_to_power d              NL23::hydrogen_to_power e              NL21::battery   f              NL23::battery   g              NL31::hydrogen_to_power h              NL32::hydrogen_to_power i              NL32::battery   j              NL42::hydrogen_to_power k              NL12::battery   l              NL33::battery   m              NL11::hydrogen_to_power            W     m      W     l      W     k      W     h      W     i      W     j      W     b      W     c      W     d      W     e      W     f      W     g      W     V      W     W      W     X      W     Y      W     Z      W     [      W     \      W     ]      W     ^      W     _      W     `      W     a      �k     j      �k     i      �k     g      �k     h      �k     d      �k     e      �k     f      �k     ]      �k     ^      �k     _      �k     `      �k     a      �k     b      �k     c      �k     P      �k     Q      �k     R      �k     S      �k     T      �k     U      �k     V      �k     W      �k     X      �k     Y      �k     Z      �k     [      �k     \      �k     5      �k     6      �k     7      �k     8      �k     9      �k     :      �k     ;      �k     <      �k     =      �k     >      �k     ?      �k     @      �k     A      �k     B      �k     C      �k     D      �k     E      �k     F      �k     G      �k     H      �k     I      �k     J      �k     K      �k     L      �k     M      �k     N      �k     O   GCOL                                                                                                                                  	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5              NL22::pv_farm   6              NL33::pv_farm   7              NL11::wind_offshore     8              NL34::lost_load 9              NL42::lost_load :              NL13::pv_farm   ;              NL42::pv_farm   <              NL33::lost_load =              NL31::lost_load >              NL11::wind_onshore      ?              NL21::pv_farm   @              NL33::wind_offshore     A              NL34::wind_offshore     B              NL33::wind_onshore      C              NL41::lost_load D              NL31::pv_rooftopE              NL12::lost_load F              NL32::wind_onshore      G              NL22::pv_rooftopH              NL21::lost_load I              NL11::pv_rooftopJ              NL23::wind_offshore     K              NL23::pv_farm   L              NL42::pv_rooftopM              NL42::wind_onshore      N              NL34::pv_farm   O              NL22::lost_load P              NL12::wind_onshore      Q              NL11::lost_load R              NL12::wind_offshore     S              NL12::pv_farm   T              NL22::wind_onshore      U              NL11::pv_farm   V              NL32::lost_load W              NL32::pv_farm   X              NL33::pv_rooftopY              NL23::lost_load Z              NL13::lost_load [              NL32::pv_rooftop\              NL21::wind_onshore      ]              NL21::pv_rooftop^              NL13::wind_onshore      _              NL23::pv_rooftop`              NL31::pv_farm   a              NL34::pv_rooftopb              NL13::pv_rooftopc              NL41::wind_onshore      d              NL12::pv_rooftope              NL41::pv_rooftopf              NL34::wind_onshore      g              NL32::wind_offshore     h              NL31::wind_onshore      i              NL23::wind_onshore      j              NL41::pv_farm   k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �              NL32::hvac_transmission:NL33    �              NL32::hvac_transmission:NL23    �              NL22::hvac_transmission:NL41    �              NL33::hvac_transmission:NL41    �              NL33::hvac_transmission:NL32    �              NL32::hvac_transmission:NL31    �              NL41::hvac_transmission:NL33    �              NL13::hvac_transmission:NL11    �              NL12::hvac_transmission:NL11    �              NL41::hvac_transmission:NL22    �              NL31::hvac_transmission:NL33    �              NL12::hvac_transmission:NL23    �              NL21::hvac_transmission:NL23    �              NL31::hvac_transmission:NL32    �              NL33::hvac_transmission:NL31    �              NL34::hvac_transmission:NL41    �              NL11::hvac_transmission:NL13    �              NL41::hvac_transmission:NL42    �              NL21::hvac_transmission:NL13       �}     	      �}           �}           �}           �}           �}           �}           �k     �      �k     �      �k     �      �k     �      �k     �      �}           �}           �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �      �k     �   GCOL                        NL22::hvac_transmission:NL21                  NL41::hvac_transmission:NL34                  NL23::hvac_transmission:NL12                  NL42::hvac_transmission:NL41                  NL21::hvac_transmission:NL22                  NL11::hvac_transmission:NL12                  NL23::hvac_transmission:NL32                  NL23::hvac_transmission:NL21    	              NL13::hvac_transmission:NL21    
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL32::wind_offshore     �              NL12::pv_rooftop�              NL32::hydrogen_to_power �              NL34::wind_onshore      �              NL21::pv_rooftop�              NL13::wind_onshore      �       
       NL13::ccgt      �              NL23::hvac_transmission:NL21    �              NL12::hydrogen_to_power �              NL34::pv_rooftop�              NL33::hvac_transmission:NL31    �              NL23::hvac_transmission:NL12    �              NL34::hvac_transmission:NL41    �              NL23::hydrogen_to_power �              NL21::battery   �              NL21::curtailment       �              NL11::hvac_transmission:NL13    �              NL13::pv_rooftop�              NL41::wind_onshore      �       
       NL22::ccgt      �              NL11::lost_load �              NL11::pv_farm   �              NL32::lost_load �              NL33::pv_rooftop�              NL22::hvac_transmission:NL21    �              NL32::pv_rooftop�              NL34::hydrogen_to_power �              NL21::wind_onshore      �              NL21::hvac_transmission:NL23    �              NL11::battery   �              NL33::pv_farm   �              NL33::hydrogen_to_power �              NL34::lost_load �              NL33::curtailment       �              NL42::lost_load �              NL22::curtailment       �              NL22::hvac_transmission:NL41    �              NL42::pv_farm   �              NL31::curtailment       �              NL31::lost_load �               �                  ƕ     Z      ƕ     Y   
   ƕ     X      ƕ     V      ƕ     W      ƕ     R      ƕ     S      ƕ     T      ƕ     U      ƕ     J      ƕ     K      ƕ     L      ƕ     M      ƕ     N      ƕ     O      ƕ     P      ƕ     Q   
   ƕ     :      ƕ     ;      ƕ     <      ƕ     =      ƕ     >      ƕ     ?      ƕ     @      ƕ     A      ƕ     B      ƕ     C   
   ƕ     D      ƕ     E      ƕ     F      ƕ     G      ƕ     H   
   ƕ     I   
   ƕ           ƕ        
   ƕ           ƕ        
   ƕ           ƕ           ƕ            ƕ     !      ƕ     "      ƕ     #      ƕ     $      ƕ     %   
   ƕ     &      ƕ     '      ƕ     (      ƕ     )      ƕ     *      ƕ     +      ƕ     ,      ƕ     -      ƕ     .      ƕ     /      ƕ     0      ƕ     1      ƕ     2      ƕ     3      ƕ     4      ƕ     5      ƕ     6      ƕ     7      ƕ     8      ƕ     9      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �   
   �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �   
   �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      �}     �      ƕ           ƕ           ƕ           ƕ           ƕ        
   ƕ           ƕ           ƕ           ƕ     	      ƕ     
      ƕ        
   ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ           ƕ        GCOL                        NL42::wind_onshore                    NL33::hvac_transmission:NL32                  NL11::wind_onshore                    NL21::pv_farm                 NL32::curtailment              
       NL11::ccgt                    NL34::wind_offshore                   NL33::wind_onshore      	              NL31::battery   
              NL31::pv_rooftop              NL12::lost_load        
       NL34::ccgt                    NL32::hvac_transmission:NL31                  NL22::pv_rooftop              NL41::hvac_transmission:NL33                  NL22::hydrogen_to_power               NL11::curtailment                     NL11::pv_rooftop              NL13::battery                 NL31::hvac_transmission:NL33                  NL12::hvac_transmission:NL23                  NL23::pv_farm                 NL31::pv_farm                 NL23::curtailment                     NL34::pv_farm          
       NL42::ccgt                    NL41::hvac_transmission:NL34           
       NL12::ccgt                    NL22::pv_farm          
       NL32::ccgt                    NL21::hydrogen_to_power                NL32::hvac_transmission:NL33    !              NL32::hvac_transmission:NL23    "              NL11::wind_offshore     #              NL33::hvac_transmission:NL41    $              NL13::pv_farm   %              NL33::lost_load &       
       NL33::ccgt      '              NL41::battery   (              NL13::hydrogen_to_power )              NL42::battery   *              NL33::wind_offshore     +              NL41::lost_load ,              NL13::hvac_transmission:NL11    -              NL12::hvac_transmission:NL11    .              NL32::wind_onshore      /              NL12::curtailment       0              NL21::lost_load 1              NL41::hydrogen_to_power 2              NL23::wind_offshore     3              NL41::hvac_transmission:NL22    4              NL13::curtailment       5              NL42::pv_rooftop6              NL31::hvac_transmission:NL32    7              NL22::lost_load 8              NL33::battery   9              NL13::hvac_transmission:NL21    :       
       NL41::ccgt      ;              NL12::wind_onshore      <              NL42::curtailment       =              NL12::wind_offshore     >              NL12::pv_farm   ?              NL23::battery   @              NL41::hvac_transmission:NL42    A              NL21::hvac_transmission:NL13    B              NL22::wind_onshore      C              NL32::pv_farm   D       
       NL31::ccgt      E              NL23::lost_load F              NL31::hydrogen_to_power G              NL13::lost_load H              NL22::battery   I       
       NL23::ccgt      J              NL42::hydrogen_to_power K              NL34::battery   L              NL34::curtailment       M              NL11::hvac_transmission:NL12    N              NL23::hvac_transmission:NL32    O              NL21::hvac_transmission:NL22    P              NL42::hvac_transmission:NL41    Q              NL41::curtailment       R              NL31::wind_onshore      S              NL41::pv_rooftopT              NL32::battery   U              NL23::pv_rooftopV              NL23::wind_onshore      W              NL12::battery   X       
       NL21::ccgt      Y              NL11::hydrogen_to_power Z              NL41::pv_farm   [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �                  &�     �      &�     �   
   &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �   
   &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �   
   &�     �      &�     �      &�     �      &�     �      &�     �   
   &�     p   
   &�     q   
   &�     r      &�     s   
   &�     t      &�     u      &�     v      &�     w      &�     x      &�     y   
   &�     z      &�     {      &�     |      &�     }      &�     ~      &�           &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     =      &�     >      &�     ?      &�     @      &�     A      &�     B   
   &�     C      &�     D      &�     E      &�     F      &�     G      &�     H      &�     I      &�     J   
   &�     K      &�     L      &�     M      &�     N      &�     O      &�     P      &�     Q      &�     R      &�     S      &�     T      &�     U      &�     V      &�     W      &�     X      &�     Y      &�     Z      &�     [      &�     \      &�     ]      &�     ^      &�     _   
   &�     `      &�     a      &�     b      &�     c      &�     d      &�     e   
   &�     f      &�     g      &�     h      &�     i      &�     j      &�     k      &�     l      &�     m      &�     n      &�     o   GCOL                                                                                                                                   	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =              NL32::wind_offshore     >              NL12::pv_rooftop?              NL32::hydrogen_to_power @              NL34::wind_onshore      A              NL21::pv_rooftopB              NL13::wind_onshore      C       
       NL13::ccgt      D              NL12::hydrogen_to_power E              NL34::pv_rooftopF              NL23::hydrogen_to_power G              NL21::battery   H              NL21::curtailment       I              NL13::pv_rooftopJ              NL41::wind_onshore      K       
       NL22::ccgt      L              NL11::lost_load M              NL11::pv_farm   N              NL32::lost_load O              NL33::pv_rooftopP              NL32::pv_rooftopQ              NL34::hydrogen_to_power R              NL21::wind_onshore      S              NL11::battery   T              NL33::pv_farm   U              NL33::hydrogen_to_power V              NL34::lost_load W              NL33::curtailment       X              NL42::lost_load Y              NL22::curtailment       Z              NL42::pv_farm   [              NL31::curtailment       \              NL31::lost_load ]              NL11::wind_onshore      ^              NL21::pv_farm   _              NL32::curtailment       `       
       NL11::ccgt      a              NL34::wind_offshore     b              NL33::wind_onshore      c              NL31::battery   d              NL31::pv_rooftope              NL12::lost_load f       
       NL34::ccgt      g              NL22::pv_rooftoph              NL22::hydrogen_to_power i              NL11::curtailment       j              NL11::pv_rooftopk              NL13::battery   l              NL23::pv_farm   m              NL31::pv_farm   n              NL42::wind_onshore      o              NL34::pv_farm   p       
       NL23::ccgt      q       
       NL42::ccgt      r       
       NL12::ccgt      s              NL22::pv_farm   t       
       NL32::ccgt      u              NL21::hydrogen_to_power v              NL23::curtailment       w              NL11::wind_offshore     x              NL13::pv_farm   y              NL33::lost_load z       
       NL33::ccgt      {              NL41::battery   |              NL13::hydrogen_to_power }              NL42::battery   ~              NL33::wind_offshore                   NL41::lost_load �              NL32::wind_onshore      �              NL12::curtailment       �              NL21::lost_load �              NL41::hydrogen_to_power �              NL23::wind_offshore     �              NL13::curtailment       �              NL42::pv_rooftop�              NL22::lost_load �              NL33::battery   �       
       NL41::ccgt      �              NL12::wind_onshore      �              NL42::curtailment       �              NL12::wind_offshore     �              NL12::pv_farm   �              NL23::battery   �              NL22::wind_onshore      �              NL32::pv_farm   �       
       NL31::ccgt      �              NL23::lost_load �              NL31::hydrogen_to_power �              NL13::lost_load �              NL22::battery   �              NL32::battery   �              NL23::pv_rooftop�              NL42::hydrogen_to_power �              NL34::battery   �              NL34::curtailment       �              NL41::curtailment       �              NL12::battery   �              NL31::wind_onshore      �              NL41::pv_rooftop�       
       NL21::ccgt      �              NL23::wind_onshore      �              NL11::hydrogen_to_power �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL33::curtailment       �              NL22::curtailment       �              NL41::demand_electricity�              NL22::demand_electricity�              NL31::curtailment       �              NL32::curtailment       �              NL31::demand_electricity�              NL32::demand_electricity�              NL12::curtailment       �              NL11::curtailment       �              NL13::curtailment       �              NL23::curtailment       �              NL34::curtailment       �              NL21::curtailment       �              NL41::curtailment       �              NL42::curtailment       �              NL23::demand_electricity�              NL34::demand_electricity�              NL13::demand_electricity�              NL11::demand_electricity�              NL12::demand_electricity�              NL33::demand_electricity�              NL42::demand_electricity�              NL21::demand_electricity�               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �                                                                                                                                              	              
                                                                                                                                                                                                                                                                                                                                   !              "              #              $              %              &              '              (              )              *              +              ,              -              .              /              0              1              2              3              4              5              6              7              8              9              :              ;              <              =              >              ?              @              A              B              C              D              E              F              G              H              I              J              K              L              M              N              O              P              Q              R              S              T              U              V              W              X              Y              Z              [              \              ]              ^              _              `                 &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      &�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     F     `�     E     `�     D     `�     B     `�     C     `�     =     `�     >     `�     ?     `�     @     `�     A     `�     3     `�     4     `�     5     `�     6     `�     7     `�     8     `�     9     `�     :     `�     ;     `�     <     `�           `�     !     `�     "     `�     #     `�     $     `�     %     `�     &     `�     '     `�     (     `�     )     `�     *     `�     +     `�     ,     `�     -     `�     .     `�     /     `�     0     `�     1     `�     2     `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�           `�          `�          `�          `�          `�          `�          `�          `�          `�     	     `�     
     `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�          `�       
   `f     p   
   `f     o   
   `f     n   
   `f     k   
   `f     l   
   `f     m   
   `f     e   
   `f     f   
   `f     g   
   `f     h   
   `f     i   
   `f     j      ��     �      ��     �   OCHK    ��     �      4       _Netcdf4Dimid                          �SP�   `�     �      `�     �   
   `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�           `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     �      `�     m   
   `�     n      `�     o      `�     p      `�     q      `�     r      `�     s      `�     t      `�     u      `�     v      `�     w      `�     x   
   `�     y      `�     z      `�     {      `�     |      `�     }   
   `�     ~   
   `�     J      `�     K   
   `�     L      `�     M   
   `�     N      `�     O      `�     P      `�     Q      `�     R      `�     S      `�     T      `�     U      `�     V      `�     W      `�     X   
   `�     Y      `�     Z      `�     [      `�     \      `�     ]      `�     ^      `�     _      `�     `      `�     a      `�     b      `�     c      `�     d      `�     e      `�     f      `�     g      `�     h      `�     i      `�     j      `�     k      `�     l      `�           `�           `�           `�           `�           `�           `�     	      `�     
      `�           `�        
   `�           `�           `�           `�           `�           `�           `�           `�           `�           `�           `�           `�           `�           `�        
   `�           `�           `�           `�           `�           `�            `�     !      `�     "      `�     #      `�     $      `�     %      `�     &      `�     '      `�     (      `�     )      `�     *      `�     +      `�     ,      `�     -      `�     .      `�     /      `�     0      `�     1      `�     2      `�     3      `�     4      `�     5   
   `�     6      `�     7      `�     8      `�     9      `�     :      `�     ;      `�     <   
   `�     =      `�     >      `�     ?      `�     @      `�     A      `�     B      `�     C      `�     D      `�     E      `�     F      `�     G      `�     H      `�     I   GCOL    @                                                  NL13::hvac_transmission:NL21                  NL21::demand_electricity              NL32::wind_offshore                   NL42::demand_electricity              NL12::pv_rooftop              NL32::hydrogen_to_power 	              NL34::wind_onshore      
              NL33::demand_electricity              NL21::pv_rooftop              NL13::wind_onshore             
       NL13::ccgt                    NL23::hvac_transmission:NL21                  NL12::demand_electricity              NL12::hydrogen_to_power               NL34::pv_rooftop              NL23::hvac_transmission:NL12                  NL33::hvac_transmission:NL31                  NL34::hvac_transmission:NL41                  NL23::hydrogen_to_power               NL21::battery                 NL21::curtailment                     NL11::hvac_transmission:NL13                  NL13::pv_rooftop              NL41::wind_onshore             
       NL22::ccgt                    NL11::lost_load               NL11::pv_farm                 NL32::lost_load               NL33::pv_rooftop               NL22::hvac_transmission:NL21    !              NL32::pv_rooftop"              NL34::hydrogen_to_power #              NL21::wind_onshore      $              NL21::hvac_transmission:NL23    %              NL34::demand_electricity&              NL11::battery   '              NL33::pv_farm   (              NL33::hydrogen_to_power )              NL34::lost_load *              NL33::curtailment       +              NL42::lost_load ,              NL22::curtailment       -              NL22::hvac_transmission:NL41    .              NL32::hvac_transmission:NL31    /              NL42::pv_farm   0              NL31::curtailment       1              NL31::lost_load 2              NL33::hvac_transmission:NL32    3              NL11::wind_onshore      4              NL21::pv_farm   5              NL32::curtailment       6       
       NL11::ccgt      7              NL34::wind_offshore     8              NL31::demand_electricity9              NL33::wind_onshore      :              NL31::battery   ;              NL12::lost_load <              NL31::pv_rooftop=       
       NL34::ccgt      >              NL22::pv_rooftop?              NL41::hvac_transmission:NL33    @              NL22::hydrogen_to_power A              NL11::curtailment       B              NL11::pv_rooftopC              NL13::battery   D              NL31::hvac_transmission:NL33    E              NL12::hvac_transmission:NL23    F              NL23::pv_farm   G              NL31::pv_farm   H              NL42::wind_onshore      I              NL34::pv_farm   J       
       NL42::ccgt      K              NL41::hvac_transmission:NL34    L       
       NL12::ccgt      M              NL22::pv_farm   N       
       NL32::ccgt      O              NL21::hydrogen_to_power P              NL32::hvac_transmission:NL33    Q              NL23::curtailment       R              NL11::wind_offshore     S              NL32::hvac_transmission:NL23    T              NL33::hvac_transmission:NL41    U              NL41::demand_electricityV              NL22::demand_electricityW              NL13::pv_farm   X              NL33::lost_load Y       
       NL33::ccgt      Z              NL41::battery   [              NL13::hydrogen_to_power \              NL42::battery   ]              NL33::wind_offshore     ^              NL41::lost_load _              NL13::hvac_transmission:NL11    `              NL12::hvac_transmission:NL11    a              NL32::demand_electricityb              NL32::wind_onshore      c              NL12::curtailment       d              NL21::lost_load e              NL41::hydrogen_to_power f              NL23::wind_offshore     g              NL41::hvac_transmission:NL22    h              NL13::curtailment       i              NL42::pv_rooftopj              NL31::hvac_transmission:NL32    k              NL22::lost_load l              NL33::battery   m              NL41::curtailment       n       
       NL41::ccgt      o              NL12::wind_onshore      p              NL42::curtailment       q              NL12::wind_offshore     r              NL12::pv_farm   s              NL23::battery   t              NL41::hvac_transmission:NL42    u              NL23::demand_electricityv              NL22::wind_onshore      w              NL21::hvac_transmission:NL13    x              NL32::pv_farm   y       
       NL31::ccgt      z              NL23::lost_load {              NL31::hydrogen_to_power |              NL13::lost_load }              NL22::battery   ~       
       NL23::ccgt                    NL13::demand_electricity�              NL11::demand_electricity�              NL34::battery   �              NL34::curtailment       �              NL42::hydrogen_to_power �              NL11::hvac_transmission:NL12    �              NL23::hvac_transmission:NL32    �              NL42::hvac_transmission:NL41    �              NL21::hvac_transmission:NL22    �              NL31::wind_onshore      �              NL41::pv_rooftop�              NL32::battery   �              NL23::pv_rooftop�              NL23::wind_onshore      �              NL12::battery   �       
       NL21::ccgt      �              NL11::hydrogen_to_power �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL41::battery   �              NL42::battery   �              NL21::battery   �              NL31::battery   �              NL23::battery   �              NL13::battery   �              NL11::battery   �              NL32::battery   �              NL34::battery   �              NL12::battery   �              NL33::battery   �              NL22::battery   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL31::pv_farm   �              NL23::curtailment       �              NL42::wind_onshore      �              NL34::pv_farm   �              NL23::wind_onshore      �              NL41::pv_rooftop�              NL13::demand_electricity              NL41::curtailment                    NL12::wind_onshore                   NL12::wind_offshore                  NL12::pv_farm                NL41::demand_electricity             NL22::demand_electricity             NL13::pv_farm                NL33::lost_load              NL32::wind_onshore      	             NL21::lost_load 
             NL23::wind_offshore                  NL13::curtailment                    NL42::pv_rooftop             NL32::wind_offshore                  NL12::demand_electricity             NL34::pv_rooftop             NL21::curtailment                    NL11::pv_farm                NL32::lost_load              NL33::pv_rooftop             NL32::pv_rooftop             NL34::lost_load              NL33::curtailment                    NL22::curtailment                    NL42::pv_farm                NL31::curtailment                    NL32::curtailment                    NL34::wind_offshore                  NL33::wind_onshore                   NL31::pv_rooftop             NL22::pv_rooftop             NL11::pv_rooftop              NL12::pv_rooftop!             NL34::wind_onshore      "             NL33::demand_electricity#             NL21::pv_rooftop$             NL13::wind_onshore      %             NL13::pv_rooftop&             NL41::wind_onshore      '             NL11::lost_load (             NL21::wind_onshore      )             NL34::demand_electricity*             NL33::pv_farm   +             NL42::lost_load ,             NL31::lost_load -             NL11::wind_onshore      .             NL21::pv_farm   /             NL31::demand_electricity0             NL12::lost_load 1             NL11::curtailment       2             NL23::pv_farm   3             NL13::lost_load 4             NL22::pv_farm   5             NL11::wind_offshore     6             NL33::wind_offshore     7             NL41::lost_load 8             NL32::demand_electricity9             NL12::curtailment       :             NL22::lost_load ;             NL21::demand_electricity<             NL42::demand_electricity=             NL42::curtailment       >             NL23::demand_electricity?             NL22::wind_onshore      @             NL32::pv_farm   A             NL23::lost_load B             NL11::demand_electricityC             NL34::curtailment       D             NL23::pv_rooftopE             NL31::wind_onshore      F             NL41::pv_farm   G              H              I              J              K              L              M              N              O              P              Q              R              S              T              U              V              W              X              Y              Z              [              \              ]              ^              _              `             NL33::curtailment       a             NL22::curtailment       b             NL41::demand_electricityc             NL22::demand_electricityd             NL31::curtailment       e             NL32::curtailment       f             NL31::demand_electricityg             NL32::demand_electricityh             NL12::curtailment       i             NL11::curtailment       j             NL13::curtailment       k             NL23::curtailment       l             NL34::curtailment       m             NL21::curtailment       n             NL41::curtailment       o             NL42::curtailment       p             NL23::demand_electricityq             NL34::demand_electricityr             NL13::demand_electricitys             NL11::demand_electricityt             NL12::demand_electricityu             NL33::demand_electricityv             NL42::demand_electricityw             NL21::demand_electricityx              y              z              {              |              }              ~                            �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �             NL22::pv_farm   �             NL33::pv_farm   �             NL11::wind_offshore     �             NL34::lost_load �             NL42::lost_load �             NL13::pv_farm   �             NL42::pv_farm   �             NL33::lost_load �             NL31::lost_load �             NL11::wind_onshore      �             NL21::pv_farm   �             NL33::wind_offshore     �             NL34::wind_offshore     �             NL33::wind_onshore      �             NL41::lost_load �             NL31::pv_rooftop�             NL12::lost_load �             NL32::wind_onshore      �             NL22::pv_rooftop�             NL21::lost_load �             NL11::pv_rooftop�             NL23::wind_offshore     �             NL23::pv_farm   �             NL42::pv_rooftop�             NL42::wind_onshore      �             NL34::pv_farm   �             NL22::lost_load �             NL12::wind_onshore      �             NL11::lost_load �             NL12::wind_offshore     �             NL12::pv_farm   �             NL22::wind_onshore      �             NL11::pv_farm   �             NL32::lost_load �             NL32::pv_farm   �             NL33::pv_rooftop�             NL23::lost_load �             NL13::lost_load �             NL32::pv_rooftop�             NL21::wind_onshore      �             NL21::pv_rooftop�             NL13::wind_onshore      �             NL23::pv_rooftop�             NL31::pv_farm   �             NL34::pv_rooftop�             NL13::pv_rooftop�             NL41::wind_onshore      �             NL12::pv_rooftop�             NL41::pv_rooftop�             NL34::wind_onshore      �             NL32::wind_offshore     �             NL31::wind_onshore      �             NL23::wind_onshore      �             NL41::pv_farm   �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �              �                                                                                                                                             	              
                                                                                                                                                                                                                                                                                                                                   !              "              #              $              %              &              '              (              )              *              +              ,              -              .              /              0              1              2              3              4              5              6              7              8              9              :              ;              <              =              >              ?              @              A              B              C              D              E              F              G              H              I              J              K              L              M              N              O              P              Q              R              S              T              U              V              W              X              Y              Z              [              \              ]              ^              _              `              a              b              c              d              e              f              g              h                 `�     w     `�     v     `�     u     `�     r     `�     s     `�     t     `�     l     `�     m     `�     n     `�     o     `�     p     `�     q     `�     `     `�     a     `�     b     `�     c     `�     d     `�     e     `�     f     `�     g     `�     h     `�     i     `�     j     `�     k     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �     `�     �      +     �       +     �   
    +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     �       +     v   
    +     w       +     x       +     y       +     z       +     {       +     |       +     }       +     ~       +            +     �       +     �   
    +     �       +     �       +     �       +     �       +     �   
    +     �   
    +     S       +     T   
    +     U       +     V   
    +     W       +     X       +     Y       +     Z       +     [       +     \       +     ]       +     ^       +     _       +     `   
    +     a       +     b       +     c       +     d       +     e       +     f       +     g       +     h       +     i       +     j       +     k       +     l       +     m       +     n       +     o       +     p       +     q       +     r       +     s       +     t       +     u       +            +            +            +            +            +            +            +            +        
    +            +            +            +            +            +            +            +            +            +            +            +             +     !       +     "   
    +     #       +     $       +     %       +     &       +     '       +     (       +     )       +     *       +     +       +     ,       +     -       +     .       +     /       +     0       +     1       +     2       +     3       +     4       +     5       +     6       +     7       +     8       +     9       +     :       +     ;       +     <       +     =       +     >   
    +     ?       +     @       +     A       +     B       +     C       +     D       +     E   
    +     F       +     G       +     H       +     I       +     J       +     K       +     L       +     M       +     N       +     O       +     P       +     Q       +     R   GCOL                                                                                                                                   	               
                                            NL21::demand_electricity              NL32::wind_offshore                   NL42::demand_electricity              NL12::pv_rooftop              NL32::hydrogen_to_power               NL34::wind_onshore                    NL33::demand_electricity              NL21::pv_rooftop              NL13::wind_onshore             
       NL13::ccgt                    NL23::hvac_transmission:NL21                  NL12::demand_electricity              NL12::hydrogen_to_power               NL34::pv_rooftop              NL23::hvac_transmission:NL12                  NL33::hvac_transmission:NL31                  NL34::hvac_transmission:NL41                  NL23::hydrogen_to_power               NL21::battery                 NL21::curtailment                      NL11::hvac_transmission:NL13    !              NL13::pv_rooftop"              NL41::wind_onshore      #       
       NL22::ccgt      $              NL11::lost_load %              NL11::pv_farm   &              NL32::lost_load '              NL33::pv_rooftop(              NL22::hvac_transmission:NL21    )              NL32::pv_rooftop*              NL34::hydrogen_to_power +              NL21::wind_onshore      ,              NL21::hvac_transmission:NL23    -              NL34::demand_electricity.              NL11::battery   /              NL33::pv_farm   0              NL33::hydrogen_to_power 1              NL34::lost_load 2              NL33::curtailment       3              NL42::lost_load 4              NL22::curtailment       5              NL22::hvac_transmission:NL41    6              NL32::hvac_transmission:NL31    7              NL42::pv_farm   8              NL31::curtailment       9              NL31::lost_load :              NL42::wind_onshore      ;              NL33::hvac_transmission:NL32    <              NL11::wind_onshore      =              NL21::pv_farm   >              NL32::curtailment       ?       
       NL11::ccgt      @              NL34::wind_offshore     A              NL31::demand_electricityB              NL33::wind_onshore      C              NL31::battery   D              NL12::lost_load E              NL31::pv_rooftopF       
       NL34::ccgt      G              NL22::pv_rooftopH              NL41::hvac_transmission:NL33    I              NL22::hydrogen_to_power J              NL11::curtailment       K              NL11::pv_rooftopL              NL13::battery   M              NL31::hvac_transmission:NL33    N              NL12::hvac_transmission:NL23    O              NL23::pv_farm   P              NL31::pv_farm   Q              NL23::curtailment       R              NL34::pv_farm   S       
       NL42::ccgt      T              NL41::hvac_transmission:NL34    U       
       NL12::ccgt      V              NL22::pv_farm   W       
       NL32::ccgt      X              NL21::hydrogen_to_power Y              NL32::hvac_transmission:NL33    Z              NL32::hvac_transmission:NL23    [              NL11::wind_offshore     \              NL33::hvac_transmission:NL41    ]              NL41::demand_electricity^              NL22::demand_electricity_              NL13::pv_farm   `              NL33::lost_load a       
       NL33::ccgt      b              NL41::battery   c              NL13::hydrogen_to_power d              NL42::battery   e              NL33::wind_offshore     f              NL41::lost_load g              NL13::hvac_transmission:NL11    h              NL12::hvac_transmission:NL11    i              NL32::demand_electricityj              NL32::wind_onshore      k              NL12::curtailment       l              NL21::lost_load m              NL41::hydrogen_to_power n              NL23::wind_offshore     o              NL41::hvac_transmission:NL22    p              NL13::curtailment       q              NL42::pv_rooftopr              NL31::hvac_transmission:NL32    s              NL22::lost_load t              NL33::battery   u              NL13::hvac_transmission:NL21    v              NL41::curtailment       w       
       NL41::ccgt      x              NL12::wind_onshore      y              NL42::curtailment       z              NL12::wind_offshore     {              NL12::pv_farm   |              NL23::battery   }              NL41::hvac_transmission:NL42    ~              NL23::demand_electricity              NL22::wind_onshore      �              NL21::hvac_transmission:NL13    �              NL32::pv_farm   �       
       NL31::ccgt      �              NL23::lost_load �              NL31::hydrogen_to_power �              NL13::lost_load �              NL22::battery   �       
       NL23::ccgt      �              NL13::demand_electricity�              NL11::demand_electricity�              NL34::battery   �              NL34::curtailment       �              NL42::hydrogen_to_power �              NL11::hvac_transmission:NL12    �              NL23::hvac_transmission:NL32    �              NL42::hvac_transmission:NL41    �              NL21::hvac_transmission:NL22    �              NL31::wind_onshore      �              NL41::pv_rooftop�              NL32::battery   �              NL23::pv_rooftop�              NL23::wind_onshore      �              NL12::battery   �       
       NL21::ccgt      �              NL11::hydrogen_to_power �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �                                                                                                                                              	              
                                                       NL31::pv_farm                NL42::wind_onshore                   NL23::curtailment                    NL34::pv_farm                NL23::wind_onshore                   NL12::battery                NL41::pv_rooftop             NL32::battery                NL13::demand_electricity             NL42::hydrogen_to_power              NL41::curtailment             
       NL41::ccgt                   NL12::wind_onshore                   NL12::wind_offshore                  NL12::pv_farm                NL23::battery         
       NL31::ccgt                   NL31::hydrogen_to_power       
       NL32::ccgt                    NL21::hydrogen_to_power !                 @R     ^      @R     ]   
   @R     [      @R     \      @R     W      @R     X      @R     Y      @R     Z      @R     P      @R     Q      @R     R      @R     S      @R     T      @R     U      @R     V   
   @R     B   
   @R     C   
   @R     D      @R     E      @R     F   
   @R     G      @R     H      @R     I      @R     J      @R     K      @R     L      @R     M      @R     N      @R     O      @R     &      @R     '      @R     (      @R     )      @R     *      @R     +      @R     ,   
   @R     -      @R     .      @R     /      @R     0      @R     1      @R     2      @R     3      @R     4      @R     5      @R     6      @R     7      @R     8      @R     9      @R     :   
   @R     ;      @R     <      @R     =      @R     >      @R     ?      @R     @      @R     A       +           +           +           +           +           +           +           +           +           +           +       
    +           +           +           +           +       
    +           +       
    +           +           @R           @R           @R           @R           @R           @R           @R           @R           @R     	      @R     
      @R           @R           @R           @R           @R           @R           @R           @R           @R        
   @R           @R           @R           @R           @R           @R           @R           @R           @R           @R           @R           @R           @R            @R     !      @R     "   
   @R     #      @R     $      @R     %   GCOL                        NL41::demand_electricity              NL22::demand_electricity              NL13::pv_farm                 NL33::lost_load               NL42::battery                 NL32::wind_onshore                    NL21::lost_load               NL41::hydrogen_to_power 	              NL23::wind_offshore     
              NL13::curtailment                     NL42::pv_rooftop              NL33::battery                 NL32::wind_offshore                   NL12::demand_electricity              NL12::hydrogen_to_power               NL34::pv_rooftop              NL23::hydrogen_to_power               NL21::battery                 NL21::curtailment              
       NL22::ccgt                    NL11::pv_farm                 NL32::lost_load               NL33::pv_rooftop              NL32::pv_rooftop              NL34::lost_load               NL33::curtailment                     NL22::curtailment                     NL42::pv_farm                 NL31::curtailment                     NL32::curtailment                     NL34::wind_offshore                    NL33::wind_onshore      !              NL31::battery   "              NL31::pv_rooftop#       
       NL34::ccgt      $              NL22::pv_rooftop%              NL11::pv_rooftop&              NL42::demand_electricity'              NL12::pv_rooftop(              NL32::hydrogen_to_power )              NL34::wind_onshore      *              NL33::demand_electricity+              NL21::pv_rooftop,              NL13::wind_onshore      -       
       NL13::ccgt      .              NL13::pv_rooftop/              NL41::wind_onshore      0              NL11::lost_load 1              NL34::hydrogen_to_power 2              NL21::wind_onshore      3              NL34::demand_electricity4              NL11::battery   5              NL33::pv_farm   6              NL33::hydrogen_to_power 7              NL42::lost_load 8              NL31::lost_load 9              NL11::wind_onshore      :              NL21::pv_farm   ;       
       NL11::ccgt      <              NL31::demand_electricity=              NL12::lost_load >              NL22::hydrogen_to_power ?              NL11::curtailment       @              NL13::battery   A              NL23::pv_farm   B       
       NL23::ccgt      C       
       NL42::ccgt      D       
       NL12::ccgt      E              NL22::pv_farm   F              NL11::wind_offshore     G       
       NL33::ccgt      H              NL41::battery   I              NL13::hydrogen_to_power J              NL33::wind_offshore     K              NL41::lost_load L              NL32::demand_electricityM              NL12::curtailment       N              NL22::lost_load O              NL21::demand_electricityP              NL42::curtailment       Q              NL23::demand_electricityR              NL22::wind_onshore      S              NL32::pv_farm   T              NL23::lost_load U              NL13::lost_load V              NL22::battery   W              NL23::pv_rooftopX              NL11::demand_electricityY              NL34::battery   Z              NL34::curtailment       [       
       NL21::ccgt      \              NL31::wind_onshore      ]              NL11::hydrogen_to_power ^              NL41::pv_farm   _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �                  `f     W      `f     V   
   `f     U      `f     S      `f     T      `f     O      `f     P      `f     Q      `f     R      `f     G      `f     H   
   `f     I      `f     J      `f     K   
   `f     L      `f     M      `f     N      `f     7      `f     8      `f     9      `f     :   
   `f     ;      `f     <      `f     =      `f     >      `f     ?      `f     @   
   `f     A      `f     B      `f     C   
   `f     D   
   `f     E      `f     F      `f        
   `f           `f        
   `f           `f           `f           `f           `f           `f           `f           `f         
   `f     !      `f     "      `f     #      `f     $      `f     %   
   `f     &      `f     '      `f     (      `f     )      `f     *      `f     +   
   `f     ,      `f     -      `f     .      `f     /      `f     0      `f     1      `f     2      `f     3      `f     4      `f     5      `f     6   GCOL                                                                                                                                  	               
                                                                                                                                                                                                  NL32::pv_rooftop       
       NL12::ccgt                    NL22::pv_farm          
       NL32::ccgt                    NL33::pv_farm                 NL11::wind_offshore                   NL34::lost_load               NL42::lost_load               NL13::pv_farm                 NL42::pv_farm                  NL33::lost_load !       
       NL33::ccgt      "              NL31::lost_load #              NL11::wind_onshore      $              NL21::pv_farm   %              NL33::wind_offshore     &       
       NL11::ccgt      '              NL34::wind_offshore     (              NL33::wind_onshore      )              NL41::lost_load *              NL31::pv_rooftop+              NL12::lost_load ,       
       NL34::ccgt      -              NL32::wind_onshore      .              NL22::pv_rooftop/              NL21::lost_load 0              NL11::pv_rooftop1              NL23::wind_offshore     2              NL23::pv_farm   3              NL42::pv_rooftop4              NL42::wind_onshore      5              NL34::pv_farm   6              NL22::lost_load 7              NL12::wind_onshore      8              NL11::lost_load 9              NL12::wind_offshore     :              NL12::pv_farm   ;       
       NL22::ccgt      <              NL22::wind_onshore      =              NL11::pv_farm   >              NL32::lost_load ?              NL32::pv_farm   @              NL33::pv_rooftopA       
       NL31::ccgt      B              NL23::lost_load C              NL13::lost_load D       
       NL42::ccgt      E       
       NL23::ccgt      F              NL21::wind_onshore      G              NL13::wind_onshore      H              NL23::pv_rooftopI       
       NL13::ccgt      J              NL31::pv_farm   K              NL34::pv_rooftopL       
       NL41::ccgt      M              NL13::pv_rooftopN              NL41::wind_onshore      O              NL12::pv_rooftopP              NL41::pv_rooftopQ              NL34::wind_onshore      R              NL21::pv_rooftopS              NL32::wind_offshore     T              NL31::wind_onshore      U       
       NL21::ccgt      V              NL23::wind_onshore      W              NL41::pv_farm   X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e       
       NL41::ccgt      f       
       NL34::ccgt      g       
       NL22::ccgt      h       
       NL13::ccgt      i       
       NL33::ccgt      j       
       NL31::ccgt      k       
       NL12::ccgt      l       
       NL11::ccgt      m       
       NL32::ccgt      n       
       NL21::ccgt      o       
       NL23::ccgt      p       
       NL42::ccgt      q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �              NL22::battery   �              NL34::hydrogen_to_power �              NL21::hydrogen_to_power �              NL11::battery   �              NL33::hydrogen_to_power �              NL41::battery   �              NL13::hydrogen_to_power �              NL42::battery   �              NL31::battery   �              NL22::hydrogen_to_power �              NL41::hydrogen_to_power �              NL13::battery   �              NL34::battery   �              NL12::battery      �w           �w     
      �w     	      �w           �w           �w           `f     �      �w           �w           �w           �w           �w           `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �      `f     �   GCOL                        NL12::hydrogen_to_power               NL23::hydrogen_to_power               NL21::battery                 NL23::battery                 NL31::hydrogen_to_power               NL32::hydrogen_to_power               NL32::battery                 NL42::hydrogen_to_power 	              NL12::battery   
              NL33::battery                 NL11::hydrogen_to_power                                                                                                                                                                                                                                                                                                                             !               "               #               $               %              NL41::battery   &              NL12::hydrogen_to_power '              NL13::hydrogen_to_power (              NL42::battery   )              NL23::hydrogen_to_power *              NL21::battery   +              NL31::battery   ,              NL23::battery   -              NL22::hydrogen_to_power .              NL41::hydrogen_to_power /              NL13::battery   0              NL31::hydrogen_to_power 1              NL11::battery   2              NL33::hydrogen_to_power 3              NL32::hydrogen_to_power 4              NL32::battery   5              NL42::hydrogen_to_power 6              NL34::battery   7              NL34::hydrogen_to_power 8              NL21::hydrogen_to_power 9              NL12::battery   :              NL33::battery   ;              NL11::hydrogen_to_power <              NL22::battery   =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V              NL41::battery   W              NL12::hydrogen_to_power X              NL13::hydrogen_to_power Y              NL42::battery   Z              NL23::hydrogen_to_power [              NL21::battery   \              NL31::battery   ]              NL23::battery   ^              NL22::hydrogen_to_power _              NL41::hydrogen_to_power `              NL13::battery   a              NL31::hydrogen_to_power b              NL11::battery   c              NL33::hydrogen_to_power d              NL32::hydrogen_to_power e              NL32::battery   f              NL42::hydrogen_to_power g              NL34::battery   h              NL34::hydrogen_to_power i              NL21::hydrogen_to_power j              NL12::battery   k              NL33::battery   l              NL11::hydrogen_to_power m              NL22::battery   n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �              NL41::battery   �              NL12::hydrogen_to_power �              NL13::hydrogen_to_power �              NL42::battery   �              NL23::hydrogen_to_power �              NL21::battery   �              NL31::battery   �              NL23::battery   �              NL22::hydrogen_to_power �              NL41::hydrogen_to_power �              NL13::battery   �              NL31::hydrogen_to_power �              NL11::battery   �              NL33::hydrogen_to_power �              NL32::hydrogen_to_power �              NL32::battery   �              NL42::hydrogen_to_power �              NL34::battery   �              NL34::hydrogen_to_power �              NL21::hydrogen_to_power �                          �w     <      �w     ;      �w     :      �w     7      �w     8      �w     9      �w     1      �w     2      �w     3      �w     4      �w     5      �w     6      �w     %      �w     &      �w     '      �w     (      �w     )      �w     *      �w     +      �w     ,      �w     -      �w     .      �w     /      �w     0      �w     m      �w     l      �w     k      �w     h      �w     i      �w     j      �w     b      �w     c      �w     d      �w     e      �w     f      �w     g      �w     V      �w     W      �w     X      �w     Y      �w     Z      �w     [      �w     \      �w     ]      �w     ^      �w     _      �w     `      �w     a      :�           :�           :�           �w     �      �w     �      `f     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      �w     �      ��     r      ��     q      ��     o      ��     p      ��     l      ��     m      ��     n      ��     e      ��     f      ��     g      ��     h      ��     i      ��     j      ��     k      ��     W      ��     X      ��     Y      ��     Z      ��     [      ��     \      ��     ]      ��     ^      ��     _      ��     `      ��     a      ��     b      ��     c      ��     d      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   OCHK    `�     �      4       _Netcdf4Dimid                       (   �rGCOL                        NL33::battery                 NL11::hydrogen_to_power               NL22::battery                                                                              	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G       
       NL12::ccgt      H              NL22::pv_farm   I       
       NL32::ccgt      J              NL33::pv_farm   K              NL11::wind_offshore     L              NL34::lost_load M              NL42::pv_rooftopN              NL42::lost_load O              NL13::pv_farm   P              NL42::pv_farm   Q              NL33::lost_load R       
       NL33::ccgt      S              NL31::lost_load T              NL11::wind_onshore      U              NL21::pv_farm   V              NL33::wind_offshore     W       
       NL11::ccgt      X              NL34::wind_offshore     Y              NL33::wind_onshore      Z              NL41::lost_load [              NL31::pv_rooftop\              NL12::lost_load ]       
       NL34::ccgt      ^              NL32::wind_onshore      _              NL22::pv_rooftop`              NL21::lost_load a              NL11::pv_rooftopb              NL23::wind_offshore     c              NL23::pv_farm   d              NL31::pv_farm   e              NL42::wind_onshore      f              NL34::pv_farm   g              NL22::lost_load h              NL11::lost_load i              NL12::wind_offshore     j              NL12::pv_farm   k       
       NL22::ccgt      l              NL22::wind_onshore      m              NL11::pv_farm   n              NL32::lost_load o              NL32::pv_farm   p              NL33::pv_rooftopq       
       NL31::ccgt      r              NL23::lost_load s              NL13::lost_load t       
       NL42::ccgt      u       
       NL23::ccgt      v              NL21::wind_onshore      w              NL32::pv_rooftopx              NL13::wind_onshore      y              NL23::pv_rooftopz       
       NL13::ccgt      {              NL34::pv_rooftop|       
       NL41::ccgt      }              NL13::pv_rooftop~              NL41::wind_onshore                    NL12::wind_onshore      �              NL12::pv_rooftop�              NL41::pv_rooftop�              NL34::wind_onshore      �              NL21::pv_rooftop�              NL32::wind_offshore     �              NL31::wind_onshore      �              NL23::wind_onshore      �       
       NL21::ccgt      �              NL41::pv_farm   �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �                          :�     �   
   :�     �      :�     �      :�     �      :�     �      :�     �      :�     �      :�     �      :�     �      :�     x      :�     y   
   :�     z      :�     {   
   :�     |      :�     }      :�     ~      :�           :�     h      :�     i      :�     j   
   :�     k      :�     l      :�     m      :�     n      :�     o      :�     p   
   :�     q      :�     r      :�     s   
   :�     t   
   :�     u      :�     v      :�     w   
   :�     G      :�     H   
   :�     I      :�     J      :�     K      :�     L      :�     M      :�     N      :�     O      :�     P      :�     Q   
   :�     R      :�     S      :�     T      :�     U      :�     V   
   :�     W      :�     X      :�     Y      :�     Z      :�     [      :�     \   
   :�     ]      :�     ^      :�     _      :�     `      :�     a      :�     b      :�     c      :�     d      :�     e      :�     f      :�     g      z�     \   
   z�     [      z�     Z      z�     X      z�     Y      z�     T   
   z�     U      z�     V      z�     W      z�     L      z�     M      z�     N   
   z�     O      z�     P      z�     Q   
   z�     R   
   z�     S   
   z�     <      z�     =   
   z�     >      z�     ?      z�     @      z�     A   
   z�     B      z�     C      z�     D      z�     E      z�     F      z�     G      z�     H      z�     I      z�     J      z�     K      z�           z�           z�        
   z�           z�           z�            z�     !   
   z�     "      z�     #      z�     $      z�     %      z�     &      z�     '      z�     (      z�     )      z�     *      z�     +      z�     ,      z�     -      z�     .      z�     /      z�     0   
   z�     1      z�     2      z�     3      z�     4   
   z�     5      z�     6      z�     7      z�     8      z�     9      z�     :      z�     ;   GCOL                                                                                                                                  	               
                                                                                                                                                                                                                                                                             NL34::wind_onshore                    NL21::pv_rooftop              NL13::wind_onshore             
       NL13::ccgt                    NL34::pv_rooftop               NL13::pv_rooftop!              NL41::wind_onshore      "       
       NL22::ccgt      #              NL11::lost_load $              NL11::pv_farm   %              NL32::lost_load &              NL33::pv_rooftop'              NL32::pv_rooftop(              NL21::wind_onshore      )              NL33::pv_farm   *              NL34::lost_load +              NL42::lost_load ,              NL42::pv_farm   -              NL31::lost_load .              NL11::wind_onshore      /              NL21::pv_farm   0              NL34::wind_offshore     1       
       NL11::ccgt      2              NL33::wind_onshore      3              NL31::pv_rooftop4              NL12::lost_load 5       
       NL34::ccgt      6              NL22::pv_rooftop7              NL11::pv_rooftop8              NL23::pv_farm   9              NL31::pv_farm   :              NL42::wind_onshore      ;              NL34::pv_farm   <       
       NL12::ccgt      =              NL22::pv_farm   >       
       NL32::ccgt      ?              NL11::wind_offshore     @              NL13::pv_farm   A              NL33::lost_load B       
       NL33::ccgt      C              NL33::wind_offshore     D              NL41::lost_load E              NL32::wind_onshore      F              NL21::lost_load G              NL23::wind_offshore     H              NL42::pv_rooftopI              NL22::lost_load J              NL32::wind_offshore     K              NL12::pv_rooftopL              NL12::pv_farm   M              NL22::wind_onshore      N              NL32::pv_farm   O       
       NL31::ccgt      P              NL23::lost_load Q              NL13::lost_load R       
       NL42::ccgt      S       
       NL23::ccgt      T              NL23::pv_rooftopU       
       NL41::ccgt      V              NL12::wind_onshore      W              NL12::wind_offshore     X              NL31::wind_onshore      Y              NL41::pv_rooftopZ              NL23::wind_onshore      [       
       NL21::ccgt      \              NL41::pv_farm   ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               m               n               o               p               q               r               s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL34::wind_onshore      �              NL21::pv_rooftop�              NL13::wind_onshore      �       
       NL13::ccgt      �              NL34::pv_rooftop�              NL13::pv_rooftop�              NL41::wind_onshore      �       
       NL22::ccgt      �              NL11::lost_load �                          ��     9   
   ��     8      ��     7      ��     5      ��     6      ��     1   
   ��     2      ��     3      ��     4      ��     )      ��     *      ��     +   
   ��     ,      ��     -      ��     .   
   ��     /   
   ��     0   
   ��           ��        
   ��           ��           ��           ��        
   ��           ��            ��     !      ��     "      ��     #      ��     $      ��     %      ��     &      ��     '      ��     (      z�     �      z�     �      z�     �   
   z�     �      z�     �      z�     �      z�     �   
   z�     �      z�     �      ��           ��           ��           ��           ��           ��           ��           ��           ��     	      ��     
      ��           ��           ��        
   ��           ��           ��           ��        
   ��           ��           ��           ��           ��           ��           ��        GCOL                         NL11::pv_farm                 NL32::lost_load               NL33::pv_rooftop              NL32::pv_rooftop              NL21::wind_onshore                    NL33::pv_farm                 NL34::lost_load               NL42::lost_load 	              NL42::pv_farm   
              NL31::lost_load               NL11::wind_onshore                    NL21::pv_farm                 NL34::wind_offshore            
       NL11::ccgt                    NL33::wind_onshore                    NL31::pv_rooftop              NL12::lost_load        
       NL34::ccgt                    NL22::pv_rooftop              NL11::pv_rooftop              NL23::pv_farm                 NL31::pv_farm                 NL42::wind_onshore                    NL34::pv_farm          
       NL12::ccgt                    NL22::pv_farm          
       NL32::ccgt                    NL11::wind_offshore                   NL13::pv_farm                 NL33::lost_load        
       NL33::ccgt                     NL33::wind_offshore     !              NL41::lost_load "              NL32::wind_onshore      #              NL21::lost_load $              NL23::wind_offshore     %              NL42::pv_rooftop&              NL22::lost_load '              NL32::wind_offshore     (              NL12::pv_rooftop)              NL12::pv_farm   *              NL22::wind_onshore      +              NL32::pv_farm   ,       
       NL31::ccgt      -              NL23::lost_load .              NL13::lost_load /       
       NL42::ccgt      0       
       NL23::ccgt      1              NL23::pv_rooftop2       
       NL41::ccgt      3              NL12::wind_onshore      4              NL12::wind_offshore     5              NL31::wind_onshore      6              NL41::pv_rooftop7              NL23::wind_onshore      8       
       NL21::ccgt      9              NL41::pv_farm   :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W              NL32::hvac_transmission:NL33    X              NL32::hvac_transmission:NL23    Y              NL22::hvac_transmission:NL41    Z              NL33::hvac_transmission:NL41    [              NL33::hvac_transmission:NL32    \              NL32::hvac_transmission:NL31    ]              NL41::hvac_transmission:NL33    ^              NL13::hvac_transmission:NL11    _              NL12::hvac_transmission:NL11    `              NL41::hvac_transmission:NL22    a              NL31::hvac_transmission:NL33    b              NL12::hvac_transmission:NL23    c              NL21::hvac_transmission:NL23    d              NL31::hvac_transmission:NL32    e              NL33::hvac_transmission:NL31    f              NL34::hvac_transmission:NL41    g              NL11::hvac_transmission:NL13    h              NL41::hvac_transmission:NL42    i              NL21::hvac_transmission:NL13    j              NL22::hvac_transmission:NL21    k              NL41::hvac_transmission:NL34    l              NL23::hvac_transmission:NL12    m              NL42::hvac_transmission:NL41    n              NL21::hvac_transmission:NL22    o              NL11::hvac_transmission:NL12    p              NL23::hvac_transmission:NL32    q              NL23::hvac_transmission:NL21    r              NL13::hvac_transmission:NL21    s               t               u               v               w               x               y               z               {               |               }               ~                              �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL32::hvac_transmission:NL33    �              NL32::hvac_transmission:NL23    �              NL22::hvac_transmission:NL41    �              NL33::hvac_transmission:NL41    �              NL33::hvac_transmission:NL32    �              NL32::hvac_transmission:NL31    �              NL41::hvac_transmission:NL33    �              NL13::hvac_transmission:NL11    �              NL12::hvac_transmission:NL11    �              NL41::hvac_transmission:NL22    �              NL31::hvac_transmission:NL33    �              NL12::hvac_transmission:NL23    �              NL21::hvac_transmission:NL23    �              NL31::hvac_transmission:NL32    �              NL33::hvac_transmission:NL31    �              NL34::hvac_transmission:NL41    �              NL11::hvac_transmission:NL13    �              NL41::hvac_transmission:NL42    �              NL21::hvac_transmission:NL13    �              NL22::hvac_transmission:NL21    �              NL41::hvac_transmission:NL34    �              NL23::hvac_transmission:NL12    �              NL42::hvac_transmission:NL41    �              NL21::hvac_transmission:NL22    �              NL11::hvac_transmission:NL12    �              NL23::hvac_transmission:NL32    �              NL23::hvac_transmission:NL21    �              NL13::hvac_transmission:NL21    �               �               �               �               �               �               �               �               �               �               �               �               �               �              NL11    �              NL32    �              NL21    �              NL33    �              NL13    �              NL34    �              NL31    �              NL41    �              NL42    �              NL23    �              NL22    �              NL12    �               �               �               �              electricity     �              resource�               �               �               �               �              curtailment     �              export_electricity      �              demand_electricity      �               �               �              ccgt    �               �               �               �               �               �               �               �               �               �               �               �               �               �              battery �              import_electricity      �              curtailment     �              hydrogen_to_power       �       	       lost_load       �              wind_onshore    �              ccgt    �              export_electricity      �              pv_farm �              wind_offshore   �              demand_electricity      �       
       pv_rooftop      �               �               �               �              battery �              hydrogen_to_power       �               �               �               �               �               �               �               �               �              import_electricity      �       	       lost_load       �              wind_onshore    �              ccgt    �              pv_farm �              wind_offshore   �       
       pv_rooftop                                                                                                                                     	              
                                                       hvac_transmission:NL34               hvac_transmission:NL23               hvac_transmission:NL33               hvac_transmission:NL21               hvac_transmission:NL41               hvac_transmission:NL11               hvac_transmission:NL22               hvac_transmission:NL31               hvac_transmission:NL12               hvac_transmission:NL32               hvac_transmission:NL42               hvac_transmission:NL13                                           �        ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   
   ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �      ��     �   	   ��     �      ��     �      ��     �      ��     �   
   ��     �      ��     �      ��     �      ��     �      ��     �   	   ��     �      ��     �      ��          ��          ��          ��          ��          ��          ��          ��          ��          ��          ��          ��          ��        GCOL                        hvac_transmission                     F�                   F�                   9v                   9v                   9v                                  F�     	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W              energy_per_cap  X              energy  Y              energy_per_cap  Z              energy_per_cap  [              energy_per_cap  \              energy_per_cap  ]              energy  ^              energy  _              energy_per_cap  `              energy_per_cap  a              energy_per_cap  b              energy  c              energy  d              energy_per_cap  e              energy  f              energy_per_cap  g              energy  h              energy_per_cap  i              energy  j              energy_per_cap  k              energy_per_cap  l              energy  m              energy_per_cap  n              energy  o              energy_per_cap  p              energy  q              energy_per_cap  r              energy_per_cap  s              energy  t              energy  u              energy  v              energy_per_cap  w              energy  x              energy  y              energy_per_cap  z              energy_per_cap  {              energy_per_cap  |              energy_per_cap  }              energy_per_cap  ~              energy_per_cap                energy_per_cap  �              energy  �              energy_per_cap  �              energy_per_cap  �              energy_per_cap  �              energy_per_cap  �              energy  �              energy_per_cap  �              energy  �              energy_per_cap  �              energy  �              energy  �              energy_per_cap  �              energy_per_cap  �              energy  �              energy  �              energy  �              energy_per_cap  �              energy  �              energy_per_cap  �              energy_per_cap  �              energy_per_cap  �              energy  �              energy  �              energy  �              energy  �              energy  �              energy  �              energy  �              energy  �              energy_per_cap  �              energy_per_cap  �              energy  �              energy  �              energy  �              energy_per_cap  �              energy_per_cap  �              energy_per_cap  �              ag     �              ag     �              ag     �              �t     �              ag     �              ag     �              ag     �              ag     �              F�     �              �t     �              �t     �              �     �              �     �              r     �              �     �              �     �              r     �               OHDRM(         N       �       N       �           ?      @ 4 4�              �            �8             shuffle            deflate            M�     N   �           ������������������������P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    ��           ��        H3|OCHKP        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                ���5                         ��             ���)OHDR/         �       �           ?      @ 4 4�      �      �                   �8             shuffle            deflate            I�     �           ������������������������P        _FillValue       ?      @ 4 4�                               �X��            m�             ��            rBDnTREE  ����������������y�                              �             N       �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OCHK            �H    
   is_result              @                                P       DIMENSION_LIST                                    ��        =�a#OHDR7         N       N                                 ��                  ��        8             shuffle            deflate                 w            ������������������������H     
   is_result              @                                   em�     m�             ��            E�             �|޽OCHK            �H    
   is_result              @                                P       DIMENSION_LIST                                    ��        �Q�5OCHK            �h    t  REFERENCE_LIST       dataset                                       dimension                                                                      �            V�            �4            �"            �'            ,�            �7            �T            m�             ��            E�             ��             ^��OCHK    ��     N      P       DIMENSION_LIST                                    ��        �OCHKP        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                ���5FSSE �k       �   �   �     �   
  �     �	   ���e��2OHDR�         �       �           ?      @ 4 4�              �            �8             shuffle            deflate            S     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   PǪ�                                                x^�|�?����IBH!$B�}_C��$ٓ}ߗ�}O�o���B'[�ʾ����8����?}�������1s]�̜�k��暙��R1���Ba�7@�;�� V�I�����x �]� ҷ,_z�s1�]���ku 1�� }�[��|�a&l�|4�V(K~�u �h	�ӷ8���>/�춮��2F ��C��6���e#8^�	�I���`�� *�U��`5\3�3�RTcǓU2x����7�ƕ&�*_3��s<�X
舜azCCHEH3��5Z�[?#TvY ��I�l
��1����Y���&�d��ØM*o=y�Z���/2
qٻ�����&1�V�z�Q�nf8�"�1�����o�\s|/����:*p�{�M�L�Rfw;(/9�,�p$�G�iSd���j�iݚ��wf�ۮ����g��2�'�)y;5�+��d�Ωo�S��"K�7��?0p�ul4{��_���⠧�R����;n9������Ə#��;�
�� �-��n=X;����
�c=���#��k ʃ �O���%_����<�����Р�d�����	��g'����tp�
@�;� _8�6��<�������#�(S*�V|��o�`��l�}��L�'���A���jyC��@��~
�P�}c�0�=�^S=4Ym�?j��\�t˃ޢ;7N�$.7�_�4^c+�2�^g&B�1r����7�`��S�"SyV��k�����%�����|���Q�k���^�����#$�n��-���8�d"U�D��(���ӳ��7�w�,�jDyo��*�ZD]�\���JMHN�!r���b��ck'T���4���U�U��
hOw����$�r(����Ax)�?ĭ��n�]�]�#z���J�5�\J�w=�N��V�i�_�'����"�b�B��4��>���JP����8E%$�̭�i��^�|'��[K�'��6�~̎�.�נ9�Y�"�BGߗ�a�_��Tvk4��Ќl��2Ka~��aMCxd�=�V<5�/9#���i�{��i�t�o���(0��c�ufٴ��,�xk�2��9���NȒq�zI��~W�yl�E-:Xq�)ϼ���B�)������膈���6��5�$�ފ�
J�!��Jd�'l��b������^g*���݀塍pօU�
���*�d}0���XX��۫_�*���ވ�"z��h�T�u�=���q�:�B�\j�9�:IV����@����W�*�t&W�u�ġ�H�|-�ʀ �����f�~�*Q��N�c+^��M�� ��Z���W�+��;]��(an��츃}����uݝ3Th���h��@�h���67��^�J��gO��3��y���Na!)J��i�	dm) �k�M@c� �����9@���/@>=��aQ�y�q8�n�̂���`g�}M >���Z�Q�T��	�6 �a�qk�@���*��K{K� ��+��-��x��u�VX�nx�u��|�u,"���х"�� �>6����Z�j��B|����ޘ �ir �5�/�-��:��*/3�[0�~���z��@�.�@��A��?�J� ƺ ���ɷq��
�N����z�=F	����?9 7&�2�����0�sg?�f�UN����e$l���ㆯc��{�^QvU��������O����i3�,ʞ,I���b�*�����0�{ǿ�o��eP��6��2�j!�o��OR���#�Rz��:|=]O��!Jz����i�e{��L7"���;����+��V�,�?ﱗcӻ>�NS).G�E�j������>:�z��M��8RoD�t��~I�!��C0���L>��Uٍ@ɞ&,���x��B�ݾ�'5�\^�"�E���F�"�Fe��9!�A*�����������<��c�{Q�Xֳ(x���Uݶ�zL�}r�P|�#[Sy�Xi؈à�cr��g7h������w��.�8N8��}.���TPcP����XMIr�=�y��!���R��\��B�S̚3���K���p��a�9|6�kXj��Ѐ�:"b�˻w���-ĕ�%��sߤ�������������Z�u�0G�9�'/����)W��������*&�I��K�tah�FF���wq��W�v�8٠�3!���Ӊt�s�+�Z�{(�BC�S�'>�̵D�l�䏧��W/[(6����}�2�bv�w8T@�s���>ar��qG[d��|��(��k��J����z7���]��_7pL0Q�t�Ocb���5��=HWpבO��9��<M�E��*�i�cz�aD3)�۴j@b*qX�B�]��|_f���r�	`fq+J��{+����I����,L�Z�B�T���5�t����ī�w���tZ�SE�a���K�|sjZ�h��*�N9'��3��}m�ނ2���԰@%#��9>�Z���	E�i��ʂ��cI9x��Mt:�F3�'2i���%&TdȔәvڊڰ�8�vH&�-3�3��KV��3�����4/�	u�hM�踕ku����:	�&�#�~c��<�n���Z���J�Ow�8j����ZmΟ���S��f��,�X�W�V!��sq�߲�UN����RQJl�$�d�;��rʺ��cX'�w=߃���W �i� ����f a,��4@X4���C~�F=�7��t�4��,�0|(����>�L�m އ꧶���sE, �c��&P��ø� �D(~Ɔ�=����+�YZ �����P����_�ٗ�w�!֢��!���T\�4EV����l��ð�idVU��Ǘ7���g=�D�+e���E��>2jIb��I�x���Ƈ9�Ԃ����ݚȕ�z�}֚�5��츌�,��R)��&�Wdx���U�)F�:T�"k��%��Px���ՠ�0_�(�f��%{&1]�1�I%4�W�"��U?���t[����.b�0dEXذ;/�<(-'r��-�����A���|�F�^���0쟘Ga<��Xa�^�9c�ʼ������P���h���=��`�Ï��;|��{���Z�{9%W�������_�z_��35��\��~��ֺ��}ܭ'c����T�L�7c������́��ymq����vK�c�%3�5i���JL�J�K�t��B,�֬��!=Lt��^�$F�-YaP:+���|S��)�P�؉$L�-2N�S4�iz2����u\AK����8�ʏ>CAE�8�+w�t�Vb������`B+�:�T2����=I�C���{��������@�`�ݺ ��)����쁴/���_�d���݆��t`=�z���ʃȻ�
�A��P��Jb��L	̓J�b��U�io���<�?�[�;v4�|��Zk�V��W�=�yH�ܞ."��=��7i��!Y�3��[�w�:�W�	(P�*�Q��%�!����R��n���ΙH��n������<r��c�Y�`��qf�.MB,#/%"toG������
Zy��\�lO�C;zo;�<����Ɋ���bu�*+<�E2�C=N��7�d���R�D�#0Z�Wd��z��1ED���J�q2�-H�-N�!�f,ʹ��Q>��'D���z�sh���CԘ�vtNI����+�����Y���h|\��-����RA�ރ�Ƙ�� ��r������FB|e(N�Vf����z~ ���S��{H �����a�����I�J�q�Sr_�8S�U �b����Z���P=Q0�]��
�����y�{�@���Vhѩ�<����JyW�Z��iT�^_�������ZB�"6>���ib�]�/��{��|+�9=��{ ��\|7�ܥ���1�Ȗ2�Ĝ�")K ��I����+B;���l���M�0�_�9"����7��3���[�a|u��X�F���^ɷL�5�C��%��7z־w>�!�*���E�$o6j��{�'%������R.F�R�(�$��Ғ�V�r��`�s4���VYU�`��~�_JW�O\�y����m�t�$#�ޢ)�}���R�3:�K�s����,yߒe�fe��4�5Yt��}�'9J�H��dzrd>W���\YR�nq$2�A��`���xlFQ.ee��/��}\\���V%��)��=��C�x'�6�ɹfђ�����=D�l�N�;�C�/Wϻ���M���/Oi4WS���\ߗ��M�|����F�`�K)[)��l���'hC�;�����]����k��I�2f���:��մQeji��{-�(<���B�LyQj�C9��Z��R�s2���a���t���cf1�������8zL��ԗ���Ј�U�r{����RY	�_�fW^8˿~���w�Ԣh�Z��HO�
�F9VXJG�L�܀ݐ���G�PCǑ�����{J���2��L}�N�L��ۈ)qx}��Sd�&�m�挨��y�*Nj��۳[>���NN�D��)�~7C�!bJU�ð���2�S/�Q1���ޏ�2K!��C?f�֥�N��R�BÈO�������{�@�Q��"���}к��aX-�$�N�Cv-��L7����� ��_�:�D�Bv�~:�J<��J�����?�`��* Τ����'� �g�<�_�	�@���!�k�Wz�^��r@���<�#�Px,��ly� ��A�"�m�����Ql:gH>vЦ"S�뒐��vi<��0t��H�}�F���� LP�ȃO�~*��;��IDI�捾<q|���� ����^D�?"�O|Ł�BqS���~q�A�
�t�G|�2ՒtٕԼ��޽��%};��?�����3W|�Fye��P��������W�b'�
�;o� h���� y�����{t�q�.f�����)z���ϖ���-�kε;���߾oUّ�KWθ�#b����~(7t�&ȕ�"��,pQ��>�ޗ�~JKM���%��
[�vD��51���k��D�c��G�NJŕ"+0m�#�W��K�.@������������pӯ'�.":z�S�U�\fq_bI�fBp���Z[H`F��fb�����(a���5tF[�6�M���Z�V�ӌm{�:��5T���ǥ��Z\1����^�Uc���3�NRU\�S�A@���^L���u5���)�\'��8�Gw�j�ƕ�K��Q����3	a��ۭ
������ż��7�Q�:8P�=���X��ʌ��N��c��oA�XQ�4)t_#}���c0,P��:QC�c�4������w�T^�`�͂@{gԀ�4�3
�k���AL�q�51����}��׿Q��]r���L�+�GJ�B?�BR����xZ)�Ʊ�L�_o,������=�wP�vY%(fT�x����L�h/y��@N6������q�\S�Ma�vy� 鐗]�te^O2�ʄ�v"�ưOoD����%�Nq��������n~�9��b_�߿����	1~��܉�h�K}h�X��Ij�?L�a=�D�7(����U��}���}�/�ʇF���qb{�?8�腊YBHj�R���_`����?����
~B#����匰S�f.���?}����'���{}�qh]D�B�lX�U�1�	��
Pyae��9��w���Y	��;5����������3��H��c$���]|g-0�|7|��ľ��ә�m��`�%@�}o��ߓ��P����U�΅������a�������=gr����~Ū����}�j�B)JyIL_EbL*>kb/�����w��nȓ��{���<���*���
�@�lɝ.��Ll�YoF�9����.�����aػ.b�>�H�w��T^3���eS��܏�ya��	כ�e�ϸ��,J]~���0pN#3g���Z:�����*IA����K�ͥ�(�(͞�5B�r�|���2=vO�k�6��6Iq�yix�S�?Gϔ����<�a�~*��BL�p�9۫��a�)��eW���ڠ/kKDė+/ʑ�T��c��=��(������˲��ğnk)$6p�l0��s��!�98Ǌ��zA���@���R��"ٰ�#u����r���.Z7��w�6J�LF�s*gZ�r*{�K���	t<�°8�$�H���	!�t��q�s�hlY|E���'��Gp��vi�,+�1Tبf}l��楘�_�k�J�Z�dU(ݻ6��S~�p�_\Db��2�yf'A\c	g&�I,o�|A�sCr��K���ޟ}�S�Ԍ��w���~��m�-���w�j�����=��1a~1ł�A�aR�@����E9�F�G|��,�^����؏&ے�8v$����>}��
	�x������[s
�PZh��49�+��Ȫvhv�c���M<�����P%�f2���_����B�>��;�~���l��j#���h\jV�L��g��і�cV��ҡ�v��y�X,,�\���.�=��y.8�`^t�Z�a,-�k��bS�q��Kμ�0���g|+��'�]�AG���acf���N�M�Z_Vbm����К;i4�daooʏ���g�mĮ9��V0���d*oq��t��o�0���.27?5�\On,�z_:��N�Ǝ\��e��J9L+{���RD@uH�e�~E� ��n���i"ⶐ>��Se�B'
��9�>.r���i�E�Xj�#qo�B��>�2�u�	���Q���j
�j�,7��^���[�.�aInT�KP��hf�f���BeQ�HK₺&�؁*�ʒ6"�8u�/�����LZ��.Ud��uX��^Y�����V���lW��-�) �ME�!�-�Q6ؕX�v'��'؄��0�p������a���L4)&q�u��S0���ڥ&�}��=�"co+�8Q~h?������L�}b�iG]����K�s����a��ʽR�n�<>�{^2l$����-�-}��bƹ[g�L���߀�C)O<_�`2i~/nH���۹.�r����=�~�16�ݍy�7��Y}i)	�6�+SKc�0[*��Qs����y^:�3��lW����0��w����er'�KF1�.}?͔�ǭvUs2�gkQ���ve��d�GhcS	��h���=r��$�	��E�n��]p��W�^6v�ƏȄ^#3�l<�H��T2Vh3����[++��g-��0<�?� ;  ���SNB􆇥9��"���퇋Bw���U	�Q��,��`�%��y�@��}3�@��z�0�'p�'�g4��-�Ի��n� ��-����<K ��^M
��أ �pB��&�������P�؞�=(%#�:5^��7���
ڧ�w�|+�<��t~F��[rq�\��(�^�Ɉ����+ar�s�E�s�j2��=OAK)�?�y����:.���S?�Â��E�Z�}HR~|��"V���3�KKǇ�f0P�v�V�bO�%w�>)�ɤ]�_"E��*�C.�8 � f�jZ��־�����Al^�*�J�~^���L[�ܳ��$E�<é�V���vë��Dvco.[8/o���-�j��w8�"R"�vy���"�2�Rg۷�z���!�+�VO�#�~���vsu�Yq��/�#��3]\����W�0�_���A�&��&&�K1��J�h�z�/�@XG" 4�:�FC墵�}&�����Pʗ,`t�:O�RoP��0�B�/��$�"р����]�wZ�X�{Q9�_���z��-4�:p�W~m/(]�{���������B����Ţ��/��Q���íBI�~e���˽�Cb�f��5�dWcu��"0�m{�K��!d;��ˈ�U1eɘK���3	�6�$�fES�~s�8峬%�ܫ��_f�$��cLυ�Ǎ�z��v�bI�t�GZ'��!w,��+��ԗJ��V����j�X��Pt�s�;j��[�)eL^���㬐ߜ��u�І�7��c�u\j��n'P+7��TCO]S%6�t��-<��Mc ���Mi� �P���^��-���ޏ9���A8��L� �* ���� ����R�^�uil����C� =�0��0-@d)@�Mz �:rȓ(�%�]� ��ȥ%Y��4�J����}��נ`��� ��ס�foC����6�r(��@9��p��.�0X�}u����J��e��/v�u����ӈ�`[�>��%M�,gN�hJ��3`�s^��y��8�������*��ǭ4��(��V۷g�$�3̫i�%����������$.s��+��?qԝgO����}�%o��vU��Yϩ"��g�#��3�DS�:r��Y0Z�sN�� �;������f!�8W�%�����Y����̠�u�C��`����:���U:I���_eTQ&�)�w@͐�y�o�*�:�sg��ߏ􁱿��*�oD��B�W� �'& �N� N��Crؐ}����+7���7B�C�LFfʃY8P�|�V�w!~dV��\U��8U�ɾj��V�M��5y ��Ua�FQ
4�8�U� ��"��H�#h�-IAE�&P9����k�m���~·�7w�G}'�7�mJ�Le5٘B_��zR�:$�"�x
729�lTVq �~�@�2�Ќ9��o~�1�7{(�{��kK4���_�_�;.���,3�����#������Vo��xh �801$}�����4Ӫ�s��C
�l�k�C��C֩��,%�޵�mx�.�S��1��	����A��k��K+^}!�[K锾�L�@qW��R&*@3ďT�k���C�o����mԜ��E��xI�,c������C�&�G~-��)�t}z���M����q����]���v�[9��J�Z�)�vU����ӭa/�彺�������6�����(�����򜼗7H,V*�+����@�}��ؓ��%�w];��Er�_r�+�����:׆��S�]�)��M��\xk��sGoj�����{<���^՗��u5��P2���[�L�_�7����ߣ�x���Ks�yT֑b�\���:�5Y�Y�5)ҝ˷�G����9�\ҙ�9��5)��8sɶ��`���_��Z�]�t�%7φ=h�3q�r�Nc�u7�q6�7��hÉ\A��z�v��܎T��]������F�km��첆�:���G��+Ͻ�|w�o7O�,��b�������	7�o*ߞ�U�|���W��w���P��ؕW����ч��)1��ҨP�!�C�¡(�/�n��X{ص�"�$�YV9�~|V\!����V�X�p�yᢢ�q�ߟ�7M���"dF*�� M@�AϬ�C!{�z:��E؉|6�O�� vx�*�KcCx"��(����v���_��-��2@9�o�r�V- ���~ �J��� {�]�P�OD��YE�b��� ��*Xh��^�||�](Bj���z���FK ��HN�O�
�/d�' [� " �2�8=?	�� CB�nC~���gP����˩��Vu�T��&
>LI0z߫��V�q��עj������L�B��B�*���'�և� צP�o	�R�v�3��g��*Bzj��K��Τ0�`ke0Z�x�/&X�S�p��ioϷ��;"+��)_�(YNIL{��wR��-a�.�{�٢�t%�3"I.���,�γ�I���e�Q/���8���w��*�s��}���תД�/IVvA������a�I�\Ek������}.��IEMg������Z�7�lu��9&;�`j~���T�������41�X��v���:�K���UΟdC��vU��~4�e��w���H�8ݛ׶^\���d��]��>i�@X*{O��N�1ј'b[~�kT��kj|biZ5��뛯�K��l��~8�%��;k���Ll��dRڹ�ds���4�Ѱ�e�M����IW3�Ur�|D[h^\��hL�O���^w�	�7@�"�}[[�wM��z����N�[�Ic��'K�:~ą�?�	���H�W����:\j%|t]P}p����kn�풾_�ł4RG�MR5CX�e��Y�?�!@~�ů��������4#��:sC���s�*�����������g�_ܘ����z��G��1�Vk���~0����\�g���i�9���t)Y@�ڔ݅�s�R������v�7R�mG�ʭ��U4M�����ɜ;TxB��Pk_��}V:=%z��{�k�_�ywW�h>���V>�Z�����æދ���,��&Gz�_w�~8b��+։��<�״�ʧ��#������tsC.���r�6��G��W]%���<�9�-����!p�)�zm,�*���;����sϢ��LV��*#��_�l�X��w��֛Bht�w�a�#�)f�����I��$�6���6�E�X�U�9S*�e�`�M\�sZ+ְ'}��_���0INi��k)ldKl��g���-�Қ/��=d��4�R�ە.�_~W
m��_�#8ާ��0�?�-��J�HV3������!F�4c�I��c�+�I�w�>[{�.�7/��8�h��|g�e�RU[�LO�ҍ�|Ne�q��q�����w=����� O�xJ�#(�:��VM8 �� ?�,�#4� �a��;"W�����m��a(���>�X�3k�:��紤���Y6 g(
�N��P~-_C&� >e ?�ځ��f����Z��>�~�toQb���S��z�~���-�R�f�k1L�.��b���?M�?��#�R�~1������F�9QЗei��+'�;�u��>�6O��Ʈ�,c3��ް3�v�&���c��UV`g/�A1*���y�� �l��K��8�ƹ����k�̳[s�}��y�&B�M��΄t�M�ףO��\�3�q�O��L�A"g��e9�OE��J�ʬ��Wz���,dŏ'���j�9�#�7��ӹ��\3�S�L�$㐟�2"�|�>kټ�ģ��1xK0=W�K�D�">��ƪ�Ӿs��~�L����m5�����W�rx�͙m<�?s�8k0+v����"�ªP���> �Wl]�����%<��b`�Kb�d	�ʖˍ�B�&3b�VQ_�~�s뚲�/�Ao�^ō8�b"�Ǭ2������@:"����P���=���K�|����˰yj��Bo��z�_��b�<,~Fd���q��R�aQ�� �n8�9��fv��_}m�zB��	�^x�j��zM���?>�wO�B�MP�ؐ�B�d��͋�;��c�0����D�@��� �������'��,��DA���<x���fg ��m	��	�/񡸣G6
4������[�����%!.���ٴkU} FZ�b�S@��ף��¯��.��	�(����ɼ<�ob��b�5M��x���&�Wz�J�r?⤹��ia K�If���I�[��z�7��tS��`xfE�7��2��>/��'ʯK�u��3��I#Ӳ�і\��M���T��
ۋ9��>��h����W(�7G�r'D_[�ԅ���,�8����Z���b.�i��q#q۔��'�!������0|�w��*!��7���k��=���?��_7�송����à�`d�и ��1�J�fC����J�P|I !������JF@�
��	�'�(�<ͽ
��� G{�������q���fR6gNy�P�&P�iH����G��[�B�O	�O��a0[!�����\��~����Sr��^V���6��9{�oi���k�i�8v�iι�2�l�ƍWn2E�(�z|�Ih�;%���2�9��m�
W$5xC�p	���p�Io���BV��a�I��&V�<��g�
e�h%���Z�wA�j�#��;�wY[�̵���q$al
�G}�6���|��$�O%eX�><��������
Vk��^SI�]+�U����'nw/d.�8�{�3$�����q�l�
�9�����R�s	>V�8�E����j
O�����\_IC7���Q Y!�����aw��Oe/"�Lj���I��.ē�W�ݏ]lVo.͜�;�{���i��뱱�>�3��T������8�ǅ'U�p21gH��Zq������Q�7"c-ٖov��X	�#G�1	�KE��؞s�>���*��?7B�����b�l�^*�^7'�T-���~����O���YdO7wT8��)�:�Hh��?P�	��ixB���Q{�r��][����?xJ�p�ޚ�P�J�IQ��L$#���Ӻ��+�/�j��	r��=���E����f����5�L?N_��m���(������u�N�f�%ͮ���O4|B�_�dT�sUc��U���u��_�,|�B��x_��Sj���N�i~��4J������օ�Z/����<.Ȯ2���W�Ih�Z���,�<}z*�$f@b\zh��DÞ@��f��v�������W�4cyVZ=D�|w�8x�`��S&2�4:�j�����{@���l?���Gк�,��J��L�:Qٵ4n(~�F�C� �*T��p���^��*-r�;�(�O�������@���:ײd��&�B`��(�{���_��l&��H0�S/!ǊX՛��py����P�P�P��?�@����>��:�؊.��%�-���qӿ
�}�eKAY���~I���Zڨ��;��q���j�(�*#e����yf,!铟�ٕ��_����^WT�4������;W���w�*�o�In�/gm�E�}A���tW�YW��G��8���&>4m��:���s��ί��Wpܐ�%x��ӣ�L�WǛ��"��lN��9W��5��(5�O�U�^Y�N�������v���~|��s,�x)��>��y��������R���}�4�~s���E�iMM�oJN���h�0߶ ��F�b&�Z�<G�Rĸ��S��L������j���o�r��$���4�ȗ���zOb�ƪ�X�4���V�����H/�Hit��:u�N���8Nfw)K�w�r���G����0�ﭶg'7n�1�ST��L6[��M0�<�&.I������ښ��eqvGB�;�ͨ��cś�K��CS�Y0��o�ƻn�a~>���]���|[�o����5[��q :�����ż��	م�@���|��ӫ���B�������y��0dϪؤA=�`����L	�c��P�ҊsB���{��pH�d ��^�<ԣS0�H�Q����f�q:���(�H\������qH\z�=����W�	�+z���ܵ|i|^�]H��.i2��aH^W*�7ܶ��3�q�:�9#�	���ݫ���':V�z���#F-�މ�(��Z�ݼ�ڹ�U֪ah�;��lH,�E~�#��w�aR��T~��y�w룼Q�vi��KoM���w���N�D�b�&ھl��4�<�5Q�U(t��{���C)ե�r|؝�0ue�O��}?b��+8Ϯ��PxG���������N����m�ҍ�c�N&)?�U�e�t�?��]���5?i&�v���k5���om���3�}����ׇ8��ED{4�ϯ\ �}�/rZo��������@Fݛ@�Az��G�DT`a�,M������]-�8���	�Ka$>� �����ׂ�E�I�7z=Һ��C,B��?�)�=��� 	p��<�,t.��:鯁IE�F������>E��J!(}����C�=+�X�m�.�u����Q�/^Ѝ۠P�D��+�� �k���}[Na������Vu**�N���.� ����t'G���ϗK1c�$�)�&:p�����G��_C��\��1�
�G��[���k=��Ei��k�R>�}��3Y�@�{%�f{�=���a��ˢjJQ&KTK7�ԥh�x}0�ҧ�>t����DZH.绠��,2Ҭs��0R?��Gw�=Cd�3DQWq����+�y?�D~U������;ϮP0��ܓ�wk�E{>W�h�K����L��_o����Pu.7�"��qs��l����<�A��K
�����u�&M���W��D�>���O~�9������\.�� 2�g���&ΗW�Dc�E��]�+���,{�p�ߕ�K$-kww^l�F��&>?6�|3͚��Y��,�7y�S'&�y�23���m8�|񼃲��t|�����ĭ�`����e8��k���M{T��e��F��끋H,J��S%x+�!�����ݘlN�ڳ��x�ݱKy�TXhdb��$����8�A���u��/%'����-����k�:��f�;_�m欵6� �̀�MW�x�(狭�?z|#�!�2]��{]�����yvw����@L��ʏ�b�\[�.ʲ���\L�J���𘪻f)d�=y<Qȕ�������SI���W���ͼZ6�|#|P��4�%�>|���K���A�dw���_���#�Y�u���>E}��wq�y\�sA��"�@'����N�׭ji�L���(y�����-�\!�	���nF#JBnD�5e���S��kv%��8�T<���X��fy�j�]�f�G2U�Z������IG���Ʀ�_ʍT-!R�6Ȉ-}�춠����<}�� %zՁ@��O�5�R�e#HI���?���'���=ş�u��*Ft�*��{�}� �k��vLN��%;{�:��[�G���B7�Ջ�HGv<�I.jdVH�q��@ m�BT��˟�)�"�gq�9������_#�ո�<J'�9��O]��2��:t�E=y����]q���]�	F���.n�wXߑ�1�ش�Xt|촅ؿ�9�58����qX�c��.�m�~�vH��q��,	�����"�Y�^H��g�( �g��&��͋�Dy��guiQQ�CF
��ә�=/�����f��s��i�i���-�`{�23fu/̾��͌�n����}_�
fVŤn�����ī\�uQω.*�M巡[7�^Q�`,)�\ј��P�C��h�|�"Ͳ+?S/ay�A>�vX�a�޽Y���Zv]�o��(gP�^=��;��$����eR��ڑj6�highl�X�OG��$��H��!i�*:�2~u�H����Kh�,I���m7=��F���+�sU�h�)5�8���[m���� y��~f��X>��
Q�G<�bUQe�{s��vj��֟���x��X��3s�P�XY^���˪���A���ŷ�����pQ������ ���)WA�2�2{���qYkd��w'���b��$A��Đm��%��������i^C��<�^M{���z�;]^K�,��ܝ��qˠ�.~ ��2�Wy�s��.h�����x=-����ςÏv/�����8k�Z�|�ؽ���1J$�c����[��������G��@�rgL���V-�^�B!%�?)��~��'g���a�1�4��VI�ZM���	zG�\�� ?���d�{��&���tx%����	�ߗ�^sWJ֘��[}Y���#��g�g�t��e�Ż��4��h�����Cu�����`��d)�� �o����{�>��Fh��z˟�������q��ue\�[0'�h�F8���?�x~�����i��煌3]\⬀��@��HC(�_���`H;�/E�Bv
��k'6��ad�� a� ��ۿ!�r��>I�'$d���e�uh��l�Y6U��xQ�B�#�q
�h�� �P	
���i��A�U��!uu���>J�L:��	l�����z��������1]H��?�XtQ�cVA�ke7Fd��~!X��^J��/��y���*ZL}�%;���_�����A[S��p�)��n�m]����qǙ�w�xgR>X������as���Ww���v����z��yC���\���Sz[�\{g}~�-���ڜg{�=~)ʧ-{�,�hL��:��v���	|��P ��]]8�ZNz[6^i�ݐ7yٚ���Vo6-�7x�%$Mu#_G�Z2�l���v������U΄�F��Ҵ��ҷ>0����O����z��b�{���1���E
C�S
��ҫ�&(�hS.�m�i@�y ��,��z��:�6�#�{�4��̕�U 9/$��E�A�)66��
k��*�\��%�O�|:\4:4��/*� �Մ�K�*�� ���`�rv��b�[e��&�˛D;���1�7��mٓ)�}.ŷMUO0�O~���y���[S�fx�\�=�p�<D`i��ε�4g�.�ǰ�B���[E�x�=�2���� ���/O��/|���dC�e����`�y��|�{Ǌ���NQ�U5��}6nJY7�Ye)��_�V;	{1y�_ʁhR�mtL�w��ۙW>���4Q�oZ	��Χ�R*9.G�P(iH� �:;B�Y�?����s-Ka�j��W:�������j̜оqW��7� �h��Mh�;<��� �J����G�J(n��-1��\�D��Y%�r�Uн���0f ��lN�G��Ҏn����峡��n�G�U ݅	��O�xk�c�W�?�MІ��2TD��:|$
����Y��ҲX�[�oo�
���/o���ֹ�����;;��m��E����Җˁ��>BO{��i͋>,rSS�7�$���6e�E�2��7���9����e�[m��[��}k����M&�U7������[��b���t��K�Ȟ�c3fQܙ �ԕ�i��M�\s�N���������!|1���*^�E�,��$��v.y���5����M�qI��Tߐ*-��ȹo��^����yz3�F=��o�mɮ)���d���5
��?�0�$�[�&w��09�۹p�pǝ���ԉ!;k�I쩯=P;�I˦n3ĸC����������s��eA�ų��{ʩ���!��ȥmE����K�ڰﺞ%��?��NK�i!;���A����K�v�����ww\�ZGto�uZ�H�0�c?���R��{5�@7��uT��r�+:����������er��9\o{āI��c�z�����y3����j:N<���G?-�>�NZ��d�W����Y"����4��\=Tl#S�9��m���þ�4�fB�f�R��+>�f,�g�뽥6O�����pr��^f�/�M-�u��w��;K�����(.�/�� ڐ���5�)�*�2��\T��rR���$��x#�c����SsƢg��������"E�(��Nw���?�կT������|��n�C��k;��`/e����pg���,Gm��xl��Q��1��=W�ot�����]7��oO�g��a;�9/i���9>f��vU�?.3`�=���[/@��@l� �J}����������{���G��M"q�[��I���޲�d��	�#��d�w��$�dշ^��>�����z�s����뼮s��<�u�]
��]=�4PC��m��HH������� �{4��!;P�+v+2A>Y7TO�6:�G�]�ʐ<m�4��o���!{����x]�'��$��a����y�z��y�wT^�0�����.�6���-1�P:���s����/U� �b�qd@�H�!�Q�
���x��2�t1�[v���1�ew��y7���`�"�����;R�@zL�]�5����C�:IH���ϧ����;6����p�|2���}?9��(�ab6�7�oa�~����ͪ$���=�ٲ�ջ<�[?�>�iw��l�,�l��D-J�M���v<J�$��ˋ
��}��>��ㆸz��#���m�tո���8��8#SN�.���9��-P��r��c<$�I���ֵ��fR�J�/���U+,����:O�K�`��?����g�:p��Po%�c����~�
�6q���Lv��Z�����iB�#�l'� .�ɯ�k���?��߮��5����1-�m6L�/_�����/#}�7s�|���yd�2o02���������v�����պ,q��=���.�<�Ӧ�M۩�Ѕn���X`iK�m�\i<x7����g�:����N�;/]CO?�v^�bK͇�]�����4UP9����紋��T$���������p��(��ݵ�{����-�_�d�W6��'�V��]���g�o�i�>{��薔�kv��-��#�� ;�K�XY[6//�ӕ�3�QB��|]}b&�+^W�:�������߄��>V,�!��;���"!���C
�=L�罹���6�8k�ż�}�s������g~"����JdM�]/W�â�g]��yD�ћZ k��Ѩ�6�:s5xq�\-W�t\Р�R��q;V��R��2W��)u7q$+�9O<+ܢng�7���̹��aQ6S��u���n[��Ak.�]-3T��������n��}0�����)mz�`���F���f��3�����PlUҌ������w�A������3�u!^	��;R��'�_H�7�B���&�~��nnQ����_%�O�"os�K)A�5n�;�ف��E�� ���%���w�E]X})���x���3[��O�)�R�Cq�a�o0����y���o��a������,�Y��������ם�9J|��l��m�1x���8�Eo�Vj�Te݆�O7-���\���ǉ+^.�}�eQ(�)�ƽ�zR玕�ȹ�=������T8�$��x+�x�߫IJƌ��ר�}�Q�&����ߓ8�j Ǩ��cX<�_�0�' 'c_�k[g�XU:�#��ߠ�p�]v_��YqX�C�����Ϧ<�:��>��z����� _ۥDB�� ?�( �@8��v�����_L�Mǭ!?]��-��?�e��m����^6Y�jQ���]����$>�,<G�wX�N�		�x)���D�n2���Z�~fjQ<�2�\���|�y#C?M�'2��L@��'<3�8?��Wm��r�o��L-x��j4V|��얉>7�
����+��{��|��]/�B{�@hdb�Kf�ΧI���6'�\����p�]�<�9��]��Nڷ2�b.Qt�S�h�T}~!�xb��F��j2��_��:�6����	qu����#����t��Oޗ3�.���{%�C!R�Ձ�ѕ�;R_�Il��m�v�Z�w�I�Y-���S�9Y�����T��\����X��1Mߗ��'��-�E1�%PT�;}S�@eپ��	�o�����/8�|KR6��:�SoSN]�������������p���4�We>��^y���3"�$����֎���>ztٔ0*��LF��C�d��޸(���	��_�K�[[ď�S��P�67n��F8+�Q;��������`jB�C��:�=�U��:���_9�5�>>���'��U����_�ʀ�cXA�?=D�q���+�CP>Y���CH/�m �l��m� ���`;Y*o ���z�`���r��m�P� G����`�Bu:k���?�4tv���7�j�o�K��%J붘%��"��݉]5������fJ[V���/>������3�hK������-˼+����%tgT�*U;<�}+V�${B��2Y�.G���^q�C��w�$>S���D��4}4�k�����¦
������̐�1�U,�lVP�7=70���[�-8�d�O(��~[�
%n���>���׮�'���g�e�ѯ_���(�
뤻Ӵ������/�`�U��F�]��9�]j 7� ivj�C�X&|xpz��| �&l��v���'H}"��.;TΡ��!nƿ����Na��6�QC�y��R(�5�;�ƅ�v���� ߭�E@�RŌ	��)`��ORޝ��Ğ��\���dN�����������&~]AЬ����t~6z�v-�0#�+�Ag���v�$�#KC�b1J�'ߐ�
�ފx���=� ����'�wR7�1�Ϳ��O���$�q-����ߢ��(�	sy���G'g?�쵺�D'�'W��K��ǝr;�0���q����	;R��͛��3��Ͱbc��T�CT�P������e�p��Ee�7�3��Ϻ<����5	�����M�x�%�Vj�\�-�KH�
���L:��~<��f̿��A��ݰ�D���(�{(�D=�6�+�XDs��αU�9Z�9l�Ќ+�Z0ܛ����1zXmyz�g���a�_`���g�c���1����������v�H(�C��Gt�fޮ��)#�Q�`u�%+-�T6MO���2rG��G8%���nOV_�Ф�h0�7��w9DE��d���a\S؄[�C�2���i��Ϫos)W�nR\cQL�щLڃ�(;^3c�ŭd�g�18���)���,�*m�o�܈9�״h�4�j�3sנ7�����j�2�Z�Hس���t�6�:�_GJ`��'8r�)7����Տ��V�A��HQl �Ae��9,���H)#huH����ηդo2�Ŵ+d^yɄ�iK��۱?%�R�V�&�4KH��Њ>]�,]�^���BיB`���+,�U1���$������>�^S�A�+Vr'Vެ��)	S�9L��_;hJOG�ŎE��+W�v��dRh�3�� �Y��#Bf0�h�yk7�Ja�`��v-���� %F�xfT�A�9������to`w"��T%���3� �E�������y�M&d��0A��b'd�v�����ƙ ǲ?��#4R4B�}s�o� ]��%��o�5_v��]��X�z��Yg��	 Ϊ�����d�'7(�����¾�`����l�Κh;z��D��y�[y���{n���4��#U�)�Z�a��O�?/����v��Y� �SW#v*�):3.�P���&d�9��-%�Avv�Ίq�Z���5!��ڮ�}j�e�pA�Ƅ���
���VC7�2�"?���?��;g%���B�h��ˈ�^F�������\�����{�+�ȍ^a2t��Ki�Q�Y"~Y��ӄG��d�b�Z�m?�fx	�M׽6��/��X1�s��S=^�Iʏ��L)����|��yv�xa�q��{x�C��3��8统�Z�l�j�ilq,�r���u�ލ��rpw�~�J���l�Ҩ�?��~K�]��,����V���8�~Zoa�*�o�qyY����Ѯ � ��(e�f����[z�S�q�{1�"�E��B�������ވ���钷X�1X�V)��dc�_�<�{�d�t���^�Y�]�����.]�����2�.�sU�C`�����.�h
�q�`���M�� �A����}!�cv�@���ρ��8E�$�[S���x�7 \
١���!1���P��	C�"f�+d�4�A|�nE�����k�����
�g� ��N����^ĔX�o�,N���������_^���{F��2�$�3���&���'��	���[B�K�c�66�u��G��񸽘̧�[�mJ|���-�b�#�C�)D�o����;�֍�P5�C��	*~o���F��3Ь�%���)E�R�r`�7�F7^�~J�>2���e��0�����%���8��l�>�d����xݤg�|OU��.*ǵs8���S�k���H��i=�ݓ@&N��ז�N��k���s�K����:�җ:5�'z� � �E��=��N��[�^��Q0��s�����+T�	z:л-0�4�y��}���Ѵ�pWk�3DD�AzY�;	 wsMȐ��C���
ԧ�޶�|�8
ê&Q��W�B�È� q�`iH����o��10o��o��ٌ�0�h��n�>=�,bv�q!���e�r����*���d!��>����:W�M�\5v�^��F5M%�� �P"�֚M�fuN}�e�.v���<Br=l�f��"�9��WڣnY�ڔ��!��ſ�f̩c94&��xpx�`�0w�}�ꎯ�Z���Ճ*�g�x�*��[���S�Y�4�'O�;5����Ɖ��k3�O��>��ZN~.!M>,�&b�M*�Ar���r�` f��;�/e|�����O�9A$K�8Ծ���zS��gQ��7���9�t���Y�m��8e]�	��_�)�T��-��?Vsľ�V����TA^�D��������ֻ�V�� �t7��qE��/����GY��ُL��غ�F�� k����dqOCaWn={͕�V)�Qu������y�;,���i��P�O�9�Fz>",���rr�)e�*lo~uMI�w�n�5�A>�iJ�ۻ�[qFgmoi}F�;0I���nqo�Ș ����.#d�뻘�L7�-ׅ���\J�F&��Jx�q��A]�19�%��l��c�8"��/;(����D��VEњ�RyK,2�JA3rXq���>��&�׈��{Ӎ\p�G���b�:����2���t]-5�kev�D�z�+�l���H>̩��y�t����q�s�]���-I�h��29�8���#,u�}��4�⁭�������±_�=���q``F����g9�ju}u�ɖG��="I%��w-nv֐�̜���y\�}AQ�>{���w??�T�?��`fU���e��aJ�n�h�A�o��ड़����QM8>_�De�-�Uϱ�7ƒ3b���y�����1���7,�{��v���ŕ�����r���Hٕv�$�Q��;��m{�Z�Kv�7O��f�9X�t��1�4�ՔD��}WI�I.�B���k�3��>~`�j%|O���ڽ<�~��[���i��D)��9�N��Ji�?��R�lb�B{@��l����V���xD�
5?Ɏ����4��^�&�޶�ߦ��4���j;�p�}*+����7����p�f���ż�����/���ُ�<T���b����7}��o���+RB���vp6��v��cL'�K/%�DKwȞ󸠙r=u:z�����&L�{t;�ƓW���
��Ei}��ddP��`���s�	�����Bve�}CAx�#�o(�Wέ��dq�I�C��?0�ӤjQ���)�~vf�4������}]w����}�a�����˺OtY�n��k�:x{�D��{J�B���K܋�*�9'����K�df�������#'�#E/s;��[�[{D�d���
�"&'����+���J�ɕ�tn`�ډ��[�k\���lm���}�Aݹh���	�U�N���T��G7��ɎƳ��DX���[ēn�M��^�D�2.���K:��#�8%#^�݊S�0\BS���E���`�d�'��\��GH�g���Ɗ��Vp�oD�[���>*ȶ��V����'��k�R[��%p�q��pY���ߗ�L�!vR��\�ef����]�=�����Cr �D�Qw0&���(F��l������KG��u`18����	�iEzn�\���3�QE �Ss?���_3���R�4;?�w��tBUpbK�jȮ8�?k��F]^=�^Г?�*����W3�}y�ӻ��1��К��8Q����O���ov�h9��븴����3-���s�qH}/�\C�{(�#}w���?I�}�����6�^����|�p�|S�&�4#���K�$�0S��i�;��}�������ppUp��the�z;����B-3ӛ��0�L���qYg}$���!�>3h�T8[5�`�i��|D�Lr$$���Ml~�����_��s6���2�yB�ü2���5o��_��R�L���\�>`�6��ߍ��=X7!ܡn����t7��I�x6L?:��J���0!����~C�z�`�����.:�{U� �fݳ	�����s�a '2�S�߆���C��P�u	��{88��a�Uv?���vd����@>:,/)]�8����G���c����t�貮�`�{n��&�����{�t�:UQ	�Us��9�p�(�{I��*b�-�ba���␯��ɕ$Ü�	���JF�wӵ|�m�,:����)~��U��A���g8wkgN�b_��v�����Tsd%��?>fxi4���Mc��XF��`���3P�"a�5.���c��*���f��g��S'�\�Xu��Nm�Ov����r�9����+t̇�-2	�Ś�dH' �_E7����Ӣ3}J�~�N������Ĩ ���-����%^��@�L����W�z���	q�b��l ����(�4�5��(��1��M�3��@pA\Q��Cn�Br(D"�wDBD��U�	�sdCr�Q�P~��6���`_����x����#M��<���i �P�������A9OC�K3��ϯ��Fg��s�-�����c2!�,��\�)`�'�X}�����'��z�P)	�BC���ų5��֕^��?x���8�l�#���r��B��ԇ�`?�'�c����F��A28�tض�cmXa�����FP��e+~�-aw%�diĿ�_4F��TI��{��������F��w�kٹrP4��pw���IrΝn�Cf�l?�vZx2e1�*��spä��ת���R�o�m�J�`��t&\5�X7������҈\�n�D�qC�C��Y��A��G�l�����0E��-�C]/�����eEO`��ݐ�'�T��߆�bX�o5P��G��k���t���;m�:ѳW�#��_��\�ߺzP�(�g�@{��/�2���4�[���^���/��֑�E��+�[4>.�9�M��q_�{Da�m��hW��o�)��'�SZ�Y�����4��!��.���Y�wKm!�ӻ{�����Rﯦ_�� .�/�ɘ�05���"�)ѻۀ%K�M� :�R�w�Y̾6u�6�h�^��It��G�A-%O����K����4��d�8̤�
?�}{`}ʜ�\��7l��;+9�T�XF�o}� �\0nȐxj�~n��ui��O9��PW�G�����D����Sܴ�Qn�c[%�V�U�۷A���TT�bX���Y��M[��(N�}r	!��氚�"��F�G�Je�Sc�^)���(���u�>��a�u�6��/�Ý��3\1����v���j4�I���Y���t�c�qmm����X�<b~5_�k�+��>����EP�dO$'����w���'�ji)������oU��m�v0�jQ-ŸB�C�P�IS�,�L�t��1"�d��8�3��I�3��0��wQ���R��\nh��A;7,������i�f'+ai�#4tS�/r��'7����I�y������v�8�_#
i>r+[,i����9�i�+�Y�!���D+��*˔
�P3��KXE�oN�z��/ul̻|��֒7���~9@��Ͱ��9*^)V�$~���AG�ĭ�g,:�ůڜ���L�PB��`�\���E��ъi������I�b���=��}�M1�����ٟu��M���?�t����c���]@���~U�&���]ৠ��i������r!;��E�?���f>�{b.�!;�P�(�n���#S�ꉂC���8B���P�').$���{�5��ӂx"E|n�x~'�@�q� ^�C���F*T��1�����.��?��?��TH-`�\�E��rER�( ����Bv� ���xL&6(���������M�CHoP/�������ޒ2���`��C2LeHﵹ���J�X~�����D�5���N0�<�!9�(H�Ƨ�Ws�^��D*u��bK�K<��L��y�ᩴu�����L��?}���m��D%�c�L��/��}9���_W+��&8Ȱӧ�օ��V<R�0$Y���6�Q�������#�IcJI*�b���RD���S:-'� ,h���(��Y���f�W�w;b��Ea$itb[Ϲp[��+���ϙ��t̿j��謤������'�8;XI��*R,ر���Ҽ���c8{JW����+�������*7��	��e}�_�2q�ͪ���yF&|����c���M�U_�Ui��F��+~3zq��r�n!v��c�� }��J���j�]K;��F�v����)||t��Η��Q��.\����A���{��|��О�����W��=�n�0y��cA+��t|�w�f]S	���Z��Q��(������K!�F�귔U�k%w�_ߩ,9�a�Ҽ7&՛P�Ҩ��R5��{��N6��&��ǫA���e���n�^3B3��8�%}̃'?��ti�9��Ŗ�>8�A��<��ӚL���{�7�c��d<��2:(a~��g�yL'[r|�3�[g�+�s=-ۗ�Z��嘗C�(�&�ҷML5�L'������e`�}�O�Ȩ�N�s0Yz>\}1���r���?�58�6M�>�$	��*p��O���w ����"�k_{�N����Pz�������;�-��$�8#�
�����M�`n�hs�}�Ho�뽄 F1�-�D�U�DƧ:]7B~6)X�i��iQ�0���v|���8�����~�h���KN(���	���N����_��I�������	Q|�~04��D�"~S�[�렍��)+����H�n�x��[`��3�����@^dL�y�ؾ������Ր|�2+{�pxqvx��r�K�#����_Q����qa� ��u}���Y������KO�j;!����H����Qr��?���}�=�}���O폗����f{�.Ҷt��1W<;�5|����zJ�I����b�$��k������ ��w�8B�)� �� �>�q��CӁA����D�;�� ���q�3�C�������P�����g���99$w&���A~��P��P���Ϧ�A;����0w�_����k�֐��	�t��?<2x�
�F�:L�R��,s��~+x*�\�ɧh�y�#�)6�hs纰��T�w��a��MF]FF��}9�k?�V X�S
�o��Q�p$���+�X��Sz�W��˿s����q���&�6�_Z�c��l|��MU"(g�`�Ӝ���.�D\9�KXɲ��z{�ͺ�z
��7cyM~�f���jQ$��:�c�*F�$*�P�z��7v��5"�̼gF3qIGVh�4;ȩ�A&��qx}���@.3���%�d�A7���ep�{���Ҍ�MTS�A�_�zI�}-����^����5v��i͐ž���'��I����0yx��N\��P�1�m#�7 ����ؼ�����1{籚�)�k�/���D|p��U%�}G�U�ٝ����,�����ńo��|�òO��t}���ED2�c���xp��	=�\�����!'L;�ML���|Fp/ºGr��pI�N�����)gw���'��a�򍘷���_���5���J�&ɦ�>��W�ҳ�����WacHO�� u�!�N���x�z��-����1��/"�#�kEԿq���2�=Ί��	�2
쯠�A�!�H���r@z�� �下RHoe�@<:1�r�P���7��[�q��
���;���C@ U��:���R��c��՚�օ��>v�S�,nh'wH��n�ܺ��g�ɕU+3�U{/���Brt��|�����z���联�̫Jg�7��?:��=.Ȗ�B��J�!�Ş�c����,����d���H[�n��h���Q�d��������19)'#*G��	��޽�A��2���80I>+}�?��[�	������ǋ�фZ0\p�5�gY�JP��I��~�k���T-�^c{ď�D3O�����r������9s�/��G�uL�|�9��Q����tvp�]�è�#�������ʐ���~>�'�!�;�T��rCQI��B��b�}�M`g�x��̺� )��(�P���gg�9k0�wg�2��(��r��4����Y�z��'�M����������W�o77��q�I���Y��ި����s3�k[���'�b��:���������㷍�EDu�2R��R$�O璃_����,U�6��o�D#�\WG��?̱���K�i��_��u��t��4S�H&�5�����><�X�Q�t������(P*�^�s᥋�9�+��(*�c��g�J=K\t@�S?�En�����U��L�f��7��^�@`Z[g��j>�%��Xl���+�f��1k>5�{�o5�GǸ,;�����(���;�f���cH̿[��n&8U�/\i�ʌl+=ta�H*3,"?��E�o+
	��h܊-�{�ή�Y'�K�}X��4��[��Z�')Ѥv�?T�>2Kl�ǴU���=*3�`%��V��a�E�˴���O�D�~���F&�Mr.~�>ؑ����Q�ilG�[��������ڕ�>��u0�̳&D[tnV��,�ˊq�u��H��q��w�_�1�%W��yd(I�H�}�GX����p�wϒ��q�+�gO�S(m��}��4��=VM{5��X1~�%]��Y8��%�xlo`���Y%�4�?�3y��M����i_�\��9zU����%4g��� �"��O�3ʹc�"���k�lJh��N(~��f��)�AuZ�g���+9Sq8�C+�<����f}�a�P��wujof�:��K��s:�^����4#����P*������{
���{:��|�7+`�Bu\i�6�����W �2]��\��N��c��>'r �tm��n������O�K`��f���up2�R'tC�KQ���[:�����H0�����p�� ��j�A���~��E�}�� ���.1`�/�0�F>��TYu��K��c@6xj��1yi��'��>�n��0�O��,٫ǉ��M��\Ts^��e+vu���[cڸ\�$�-,br��]�&�O��φB)7D���ꐃX����1s�������7{��F���"��ݞ%�w�?;�C�63�-�0�)��H�f�e:�5|��0��-�'��P޲�&Cd]���b%�i)�f�ã�c��߽j�PG[.҃n)9z.>�����Z5���aN�D2�by�G����n��1��O��7�;���^��'���&h����FxL7������a0����f�\�k�sݟ��ex���TX�֦�3�+7�&�G��_pN��0bW��s>�ftg�Jz�W�6��<�Lt�vNC%���NM���U1�e΢>�6\�X��(�ov9�ߢV슗x"�:�.�����T0 �N�=^p>A�{凇ා��_���O5E��1:G��DF����;o%q|�kN����C���
9>�d����.]������.d�sU�C`Y��<	.T��R�Ǣ;{�I�=9 �P�������Au�~ݜd��q�H�i 7j���	�/?b�C���P{�S�!�	�	����ʇ�ހd�b�A|��������]��ڵ^�ՃRۂ��
��]J��ɒ����n8޼'p��J�~l��(WG���Z��[�LI.�VlU����u)�����W�VْۻH;j��m�cM6���F�
��ydȸ�+�(�Hʍ�6���	�	�ɠa��^i^ʉ�:{��F=��/����`�~��D���Ɉ��YT&�D�T;"Ĳ�SC�`I'�ݫ�ټ򮏭+��$�{Ri#<���տf�H�X�׼
h]��sr��$I~��� Q��T�xWƽs�I��9b�}u�/�+	�K�9��ׇD�y)���0�_�c(f0�z����3�+(΁�	$W:�;?0�bj砉�� ���pWk�3�ȌBzُ*�@"idH�'à��H��@on�<�od�N�@�E�]�}j��!Ε,�+��}a
��?&�Y��e�@R�ͽ�\�,�wkF��F�����Oy��	o^�)�y�/��'����j2S�k7��1�o�ͪ� �����r]Ƴ�C�S"�#�]��c�N�}��ί��%�C����K��[ض��vR#�^��'�������[K��؋L����\��O�����.�65Y�kr�$��\�Ά}�6�����O��fٴ���qGo���dWF����}��[��e�������"�	V�Ej4G�?J���^��Te��S3v��si�~���߈t��tu��+-���O���v�R��>��>��6/z�,��=jI0 �&�:�Њ�����[����l?����ȉM�?���uc'Az"�5��c[:s���[h��:����/jt'�RB���${�N��k�a����ڋ��J7���En���ޝ�~��f���|?/DVeh�Q�R�=��c��[ʚ�O���,WS��x��_s\s�)�,�K��<`��S�Nf%7��������&��WQ=�ߟS'��C��C�C��,+�"�#��ې��A-k���7�^�� ��V���c���K��S��α5��Ԉ�Ƨ���6s�0��t�ɶx'�]�"���Nx���pm�����鳱�i����?_�~��+�[���"6g�ۉE�N�Dޏ�n*�}���.�+�\��r5�P���U$���r���x�����7K��L>���qyH�[1�̬^�o����a�3�8�SA�a��C �����Z8���r�:/�SĐ3-Y��*�������-	�-�P�� ۹J\�H�����OИT�<;>�9Zׇ~�P���(`\*����܅����X�F�bA�
C��~����8*B��j�O�uN�E���eV�o��~'.��9�5�je��(���[��|9Yȉ6�Wd6Ef-�s�ߏx�>6Z6�ɓ�9�C��n};$i������r����ֳY��kg��h���oo�����Ja�O��y�%����!5�Y�m��	n����xO�0��b}r����^�U��H�7�xx����"���M�oU�Ð8G��}Kg�,��D ���@u.�ۢs�d�7��n�K�Yu���5�f�j���<�f�2�W�pD�'�F~z4�(��5H�4��yǵDj���Q:���&ٻ��e�w�$U,��ېTD�3y1r�W�,':�>��_�%�f���c�b������57m�s��tnƻ)�w��]I,*�:F�W�\���l[/�>�e%ad��mia�G���nSR�9H�.���%�}hG�ލ|����������r<�����{�,1'EeJ���5F�ފ5�̼_4֓��#����W��!?ReMc5��q�C"�56�Z�nѷa6r��?g��(����l�8��U]�����2���i3R?�Qi�'Mg�V�S�3�,����6���qx3�=-�6�[>\�"�F>����`U8=
d�����;�m[oq{)2 ���ʳ)����j[�G/!]�����Z1�2A��'��`�`��j`;I���C��;����CQ ?@��D0�o���j��wK?�����q� X�s�s��rM��^
�u��KI%!��	�8��R���H.בJ����B��х�9��}�!��60��vy�\B嘧vS:�ׇ�訉�D�u���fh�7(-�ި��;̤�3s��b�s?�@#j�ょ�޻o�������y?Q�0q�w�n_N����n|{���&
�m�B�	7�_����Ò6r9ث��gi�fIΨ7HmD@��i�a�/�����)*gz�����ְyEet=!G��/4r��b�Z%�t8�F��:�����ϴ��+�goP��~֥>��~��P�3�~\�A����So�~h�R@T�Ke�]�����B"ȳ�δvIq��K�6`����:���M�D��`�3���6��B�s���бt������1d���ԋ�<��O(.`���q�F}�DK8�@jO��H���3�Br��gPy�
���ӂS��*1��_�K��]C�`�(��r�KJ�5��������.��].���1'�������C����:f�j���5ԍwq-���Y����9��צ���_}P���z\��� Pü�B��R�w"�pC�>��q���{	��&�	v��.�#��'���N���wOB���Z�e_�I��YEթ���X�Y�����n9zO�m���_Gg����M�#EnTm���U)��I��Q	�0v��_�*f����SIr�t��l%|�D��Aځ�^����'*�T70�S��,�{T���d��Ȱ�.m|��<�~��c�h>3�\3� �K	@�4���]�K�P���%��5Ǆ���!��k�;C��
p�&RR�j��_�����Q���䓹���[��mP��6$�no�C�!	��������� .yL�;�
�rHs���Z�5���7m1�`oFU�U�jDY�N}$�OhI�bW��C�g���$!������ō3�O	p��U�=3��ʤ�eӣ��I��U�֖����	ԵE�X�`�.�)'���`e�P��k��|��d���
L]�ث*F�oxۼ[�������{��ⅦU��fLa�o/\�Ma��Wd7��]Ǽ���BH%3ax����	[�:����w�"$��M��B������v�N�d�G$��Hn���89�X�e\�B{�y�_�8V\5\V�nT��߯�J#����	�Aj��N��7ME�_�M�P?rm�H\K[Z� ~��g��X���q�Km��v3Brm<�P9r�jP�:��_���5$���	��M!~� ։m�@ؗ�r5�"�����2�z4�W4��Z�[tڪ��u��H(�S� BR�$�k4I'-=3�rL���(a�5�q���ŝI�w��5��4���t�ů��Z��"�[-&e/^xrs�K�3� 7�b1~��������\��͹s���"��r�6��W�O1���[���1�0���*B	��.���]�t�d��|����*�Nl}|�d�Ӵ���6yy:m\Vb'�\�S�}�\QSѹ�_S®�V�����_S�Ӊ����hD�{�uhԭ���W7���KEß������~�4��/��~a�lf��Nȳ#����O�v3�̄A��\�j�rM=��.�Xw�Dto8���H����̬v,1��i�ڱj�c;�ԯ7ĸ��5���?�>-��7�%��VZ�q��vL�Ԓ�I��27i��m�~��G�f���js�U��\�o�3�$�*Fk��vs=)eN�Ek�uS8�ڒ����9�#F�u�P?�����I�ְ�G����͜�g}Y����Ld�s�M�3O�=і��O��?��*���Ok'?��g(�C�.׆O�)6R䕅��J���
�`h�o%��4�x�^1�g�=�_̾ykr�K�1g*cюNߦF�v�)�`5)�fl��p� sHTϵc�T��q�v�f�o�F�
�u��p�	�h\S��`���"ҵ�n�Lu|.��K��`]N�M�_��'HZ�"'"R鸽`�L��NKP��rᶛR�9C��@��b��bY�oLW�eժx��OJ�ov�1Yyn�3tT%ϊ:C��Ӣ`)�!ε9����k�_���1.
�)j��OE^�A��ԐN���܀#=��)����Z��"�@|��#�o���1����B�����C��P~�~$��	�{�qs�{�x���Z�\�C����ސ?������Pyw�����?��M�-����#�oq����Q5lP�����|�C���B���藨1�S��A�'05P��*�[�����}ԋ� �]l�Ho�g���8=�a�!��+�`݅kc ����&����Dj����#�`���Ar+����+9-�o��8
�=��p���t�PC��$|�mv4�a�k����x�n�,��>�7��/Ћ>T���ۥ7U����L��6<��
1I7�=���fx��5!��v��U��!��;e�K�F�-����8�)����"�4�\�ڤ��od�w���S^wY^�4�O��6Pc�~c�3�Y;$c�J¦N.��gpH&ov�Ó)Vxk	W*c�Fǥ�öeE�����0�����Q�W��S�AM�x�*��5�0�����x]�O7���������x�<CIq��٭4�&C��\W�H�{�M<-�Qa�m��Y�޿X/{Ɖ��,�p���촷���R(�7j������5ْ8ʾwv�N#Oa�F)�z�G�9�i_��h`�6�,$}�4>k��+�Sj��7� �0E1�$4�����6��U]#Ն�ף�ᜍ�;�^�WVR:��Esψ���Y����v�w/��OOB��i�]|���}�e�m9|/��+0��=޳�;r���,����{��?9޳�����~����8��������h�W��#��L�w�=���\��K�?N���Nj���t-Y-Pr�G�a� Y�X�f�RM�W[��ȷ�5�O9m	�RV-��vct��:m����(*��N�,���w�mûE
�;��3�G���W���S��\P6h3c���*�%t�f�Ǔ���_(/v��k��?��ъ��Js�(��|�=���r�D�xU���U���Z'�͘8�����+TF��9wK9)�3�o�;������L}��!z7N��gߕ����.�����k=Gn���E������{��Nͦ#=p�ѸZ_T�u�0Ҽx�6���s�j�׫@ԫ,�x��Q$J�W�I��{�|��Tí�L׽��[��(���1���~`��āi$�~x������ѫ���?j�I_�����γ���0=&y���ļ�43�a:vR���"��	,�4����1f�E�[W|��T�ˎ�D���S����;���z�ؙ�C���O��x����|N
pOC�# ��&p\��p�����J�����0.�7�R�Q��1�B&	����:SP^��o� �Ƕ��y�4$W�'[f@�&�����r�(���H�v���v�!�MuJ�nXC~z��[�*���X����06�m����%��%�l^�b8g]t[��l�����������j@���YU��i��J��f��c�+ܢl���)�EDEJEDE�DE	Q��^��Z������]v�Z�����<�������u}��9�̙sΜ��^�]s��F��|��T-��a-�BVڡ�	�)\��~�\�,J.�̑���%4�$�aT����1�e�ԃ.�W�-��<?�t-�����@2��9A >��ݹ�%N앓��DW��rr1㱸]���]k�s�Ьt]��U��-��D���K���r��c��	-��G#y���{�+9�לF{�}�̰�}�|��.�L�0��G�%�WfQͤ3l<��8�)�Ȭ
����;}�]��"��l�O~�<o;]�x�0��L�������T�ٚ+�WǮφ�D&^0�T=;��֞S:���=�C��3��/���O����g�_�;z1�֡�=�S�N�i�9�n7Nٓ��A�=o�r����7�ǎEqm���������]*�����#,���wO\+���5ɠs����L
��?w����i䛠9V��$~u�|�|�i��9t� �[|��,���n�Y��
�������r[��	�<F�a��	�R����Ԁ��H�h%���П\�B|���s���uP��䯋��8�\G������z�@r�� �NӘ���:��������{ߢx��tvք�$G?}�ki07~����J�aU��������b�,.�Ul'��$��z:���rq��o�N���}�y*)�S��p9O7��SJ�o��q��R�=�,�ۗ�8v�k9��ɯVᅓ������R�Ӿ�
]|?���JDK�>�������Z�.s�B��@P�w���]\��\N�LN��̑�/�7�,��4�l�R�6<=�O��j�Tz/�(�����b���O�Vf�|��eH�/�_b�����{$>b`�XZ8/H�7�ˁq�^x���|���;��m���n i�`����)n�'�C~⭁~g�Ct �5��Ú �h����嬍V/�g��	�Pu��� 7� ��i@�r1D�/�ef���Z�@jw��^�xoy�R���΄�{x8��Z�w��_�]x�R�k*�k���R���o�>M"��f\�y�~��kI��C��?@^���~��G���H�O*�+�c�N�Xa����	'���߬Rv_�8.{������t�#�oN���a�@���]M�Z�=6�3!�����ޘ�?�B/Aq�� �c�_��{�]��o��|��]���0U�4��'e����T����I�f:��>a�X q�ۺ�kJ��k�1V�Ej.j�i�l��`�����'�w����_WK�/5�V�1}X�E)+��|`B�Ku��yJ��������}�pOw��)��8���/F��=ح����� +���L@��3.��AgI��G���b�E�֏*Մ�4�i�NI���(��l��ۡ0)����^��w�	ǻٟ�e#�晞����櫹p,;�ڧ۲y�1�v�>���I��� �/�ǐ�W�؝ ��}{��/�|?19ĵ߼K%z_v�ty���״��>1��W�mq�������ru�\T��qk�pJە��c/3C�jy÷�܂�ͳg�s�0�>���cy*/fQ#���ܨ��c2?Dr��q�a��2��b��;���qu��Ԝ���-n��R��#����
j�si��{�|l����[iыY
�F|�6S2ڈ��/D�|�ȳ8'i��F M��j��O����;�=����{�����ռ˼۾�>Vq�#�)AǠ�a�t_9����r'���w%פg`%�g��١��ɷ�aA-_H��=�\�}ě��M$�[r�Я����	N(��A��_*��J�����>u��� <��#��rG�%�+��D�7W��[��Z^����?�	���7�rMN)���\y�6'�c���7`�a�z@�� �?)M����-4]��mRty�_��ꒈ�&B�+�E�Z��,�K ��orZ�p��Z���(<j]�S4`u�. ˑ^%"��Ѡ�H;r�u���h���[�P�����O�0��~��?��o���EW��Y���N������-�{�t=��Цt9��j��&���;�׮�g�D�xM�Y�-v�]�I��ԑ���%���A1�����v��|l�'3މ=Q8�Ҥ/��ю9F|K�����W���5��gW���l��vq2(�Ʈ�mܮvjq{��;�5��7fK����4g����R9��3����'�9;��׾�;L�f�����IU+��؛�*��]�p������ǹ�LM���ao�8W�T��>����J�[HP^h�TS2�5��h�T��!���ę��ر�M�Cɂ�����-���Y�+]hB�!��o�&*��tt�݌�k��zw|��5�q��Nu������_�L�����Q��E�^���K"v�>u���a�Es��yF�����p�xֿ��+8�O��7��b��o!��\(��U����-d ���G�P������Ay��� �A���Zrܰ<��1r�p\�ia;R�"���q���0��II49T�*�[�#
���=;) ��܁\�I����k������m��Z%bA=rj� �AW@e��-7n=�h�_���_�0ߤ�]�L9���+$�[��mJ��K45H�|�y�~����A��>Y��<���w(�A��I�0�ڳ��v��zt*�U���e��K�Y5ڄ-=tph��c��2�J�]=�8�\����-�S/�`���_�����XH�U�V�-�S���ϡ��_�W�<~b�;��χ���<:y� ���_eM;e�4Ϻ����;=(%K�1�
V���ڪf$��Ҹ����t�k���46���G�~1d�|�w��C�@��tX���3�V|��n�o�ء�������	���;q��|���N��	���@=.��**܀/�t��P<�X�XXX,��:z������p�au� �c�����3���|�	�\�S\�$`�>�3�J|Tѧ>�CN(�\�Yr�\�����<��Ӌd�N�ȏf����?�'7�d�{O��ͬ�f�ˑMd�����HoM��{�^	��	Z�����gީz��]���
���ct���EhW��p�Y�6S|�<���S�ԄA�t:�hk�󕙅Ns3����Q^��+i1�ˬ-9���*-8O��!���s�#��c����Y��r[�xk�2��/c_KOY�DLsi��Y�i��\5&�I�����dQ�-�.M���Pt�c{.`��9R��o���Y<��Qh����p����w$]8&�k��s~��޽8��cPV�ۍ����E'�ݪ��vm�ͫ7I�t���Ԯ�X�r9qj��/{���.��tܼ+�j>�ܵW%6�^�AS��O�LΣM��'��^�7`t��]�
))��9�ۥѪ4~�6�U]�K�����g���h��P	m}s�w�5�����ټ�];y�e�Cm�Yc$�����E,�A�B��s�;�a��?��v	�'�R]�]�J�gX[q����U�]ú�����g��?s�c9�aՔj�wepuX���8�.��,:=[ceJ�:<��e�`,��@Ue�����8O�fz����c�DW�Yoԏ�e��=�gz���N�<ع'O���u��@|��Iq/�'����H�3��W�i�ی9�DQ���\�(�16
.��	A�|,;˾���"��Md����r�o������ǄYG��o~E��f��ܣ���*��n�Lۮ�<�빠�����o���I�9�L3B�̱�{_3E��$����1.-a.NＱ������s�ohT��V�sО�K��aR�� ���{�r���ki�_�t3	=��d=̶�^����������1N��fܾ�ۘa�ʎ��a�[�!���9��p����U��w�	�N���R?[��?3��-,�ӽ��AF�{�+��o#=�~߻3o�������S�	��g��d�K\�ʗ�6�H�R��~@0�+J:����5���H\���|�P%~��`+��0A?9��iĆ�O�/�������>OH|�(��;M�����3���n�b��j�/$���-ݱO�����ˡSt��������^,{�1��Q���J5��N��Q�1i��;��Δ��/���W����`�/_��+��\|�G.}����B��#Ir������y`yMs�oi��ޣ���n��+������{�"�#q�NT6���^Ǖ^�5H�\�H蹎�U�W5'�3�单C�f�����O��]���K������'o�7?XjqpF���fl���'Ѥ&+5��-ū��e'&�5�$����^���zL�Rq����B�`�Hmɇ\�voSd�9�)�8�[�oi��Y�3{��R�Y�Gw�B�����a�a�r�+�qT_�yRq���d�FF�F{��E��]���}r�t��Η?��y��.���od '�e�. v��?o��K/C�7+�	�E��κ=�cTJ>��oi�o�br��F��;��_M+'�<U�a��Տ�'��v	�U�9�%�\����xO9��r��D��	���!������5|�@�L�
�(���[r[m��~E����M�������=|��G��6/^���@ԓ�}�`�q:x���=q�u�9yx~V��}�aۉ׿�D������wꦍXH���J6�n$�(8r#��@��I���/5F��h������Ri�����T�L�%����W4)��X�u�O�i��Wd�DN���a|���o��7io����m�8\��K�X��~{��⛺þ���&~�ڟ~<.lI�y��Ҍt*}��gk4g��&�U��~��^1^١0 �(cɶ���a�ߍ�9¸��{�QmVa�[6;XD�쵴�4���meC��:��Qk�u�/4o�5���3��޷K ��;<�,���r��������	!�R9(��t/ L���ǃ��g�8 �S�4�ڿ!a��1�ϓ;哜�{X�]��0�W�@��y�'R��c�^�F(�ŀ|N9P��)|O�����
�U:A�������ʁ�v(��<���������l@lS���q�7���vݏY�߫�!�s�k��/6���i�݁��r�~�	=9$�d���rG��3l�F��N�}ϙnƷ�R�m��4�\�8OF.�M�j��֞����o(�cwN{R�^�H��Q�L[mFz�ǻ�u(������׎}U�(\��l����<�������%������}��r�1���/��ߣ}ٖ,xl�N��h��=�?A/��O�T�AD�N��)(�ߔ��ׁ�H��Q*�]O�����a��\���W�cC4@l��-�m���vݏ�BU8� B̐|�d�U��(�u��Z��C�<�d����@\���&�ޓP��3g��С.�:5$5L�B��h�.~�����4@�_Vpgzx.(�.*��H�����0�w^�C��z���@=j� ُ�&"�n���xv�~���ٳ|�c&�l���d�\����l���J��*|^�i�f�B��ݼ��ܮ�������uӤ�wQ�c$2��_;}���9�a����AJ�SWx�jȈ,]���p�W�u�:q}�!�-�.�+"߄�-Np�q�eJ�0; ��U�&�Z�w����W���7S�ȏ�,uZo����D)6��x;��-=�J�!�p���3Y���s���e�WQ�)�(�ɀV\w��+:����ٽ'	�Y��u�$SM��j�p�ʪ@��M!H�t
$ ]2�oX�_�G�q��J����gxP���jǧ��y�޲I�h�D@�K��e�C�J�d_�����#�y ��G �3��!���\0!�ړ 8�a�!�nXO�X�3႙񖥸�C���CJ�DL�Iܝ�ʏ�j}�+�o��u���~Јb��V�JĪ��kY����Ⱦ$��W��h����wl�'��[���\R��|�{�䭇�oŨX�;�R�4͟�Z�g$�*�hܴ=�êgiU��vNm��Y�s�ҭ��r�]g�w���?�{g�̲�J����F�;Ş�Ke���:J�3@���8��v+!?WKJޕA����Pv���w�4�M�z3��f�+*n{��C����uKu�h��ö
Q�����9�ɳ0�:8*)^�f}�Xh�����/I�}���'9v��?���~}�w�h�nN�[.>���]4�N�Q�:�Or�g���3��֗��AP%��mf�YOVX�D�Ի������g���7����FSnl��XV��x�)s����\��ˠ�3�;!���zɖ���L�?�ϭ�<�*Ǵ�����R��1��t���ݻPw�=��[��~���jjg_/��#8�\�z���A:�O�=�w.�}��t�_�k1"H�M����i΅VU����^B��B���T<�eT�g���{�B����q��܄y��0p�[z��f�S�[�]��in�rl&��)�U�U�Sy�#T-U�����X=�q~-�Ո,��h��m/���߱;�*�P��?��s�����<�T��($�.\���'(��V��vZ�=w|k�)>Z��n{�%ZG�X��]��]%��u��\��L�gm�)��>$���8�<vB��Q~���Qv�{>����E�"�/�ښ��i�2��� ��ɀ@q�9�1Z�)j��NE��~��@�k�|nhǩ!��L�v��q��Hx�B���N�>��p��3�vh��dd���C�>�q��dc�yo�}"�� P:�t0�f���h�2.��|N��$�O��7����Ơ=�'��V��vh��Y�|Qh��HA?��2|�
�S�+5���σ�����v��L$�IX/zڻ���oE���%��,`�|�]h�7����~#/^$R�a��/����"� �f�s��ȍ@�싖nE*��9?�y�2D����s��U���sR�0��>-2J9+o�����˗[�ٱ���>f�#=�ZL��b�F���)eeE�ҜU��
�����?49ɢ���Z����|]^�D�A�52r�Ύ�;p^��)�����縭��-$t	�~XF.���!'Z�X�!;����OTM��>��c�^���.ѽ���=l4s	!��R���/K�^��̕��y�v�?J��S��֙��v&��	.�!�q�����`���Z����s����%����	E7n�N7���?э�ʽ��PC[8�RҚ���J��1�p�˄3�G�\��<J�����Q:2��0C�����N���P��]	��L%�M~[Z�k:��L�k�)���/o�����Eޡ��:Xߥ��Rc�����[=�',���-��8���UH�q R��`?}<�-�O ���)�r��pHn��%�/D��OQi�}P��x���0�IX۞�i�^kJ���o��L�r��֟f\�E�X��8���!�Yn�mA�}]�~UyN�S���5���WQl��B�J���S��kȧ�GL�h����i
|/GP8g���g�o�*��v�]�?��+
���M^�O4�f�@ʂ��굽:�)�e�2C�D�`^�=��x��c�%g�i*woJ�H�l���(�<C�t��kW-��FI�4�&nZ�ϙ�\1f�=�9�,�����2�.M�=2�y%�\�;>9)�Hg\O;������tKi�M��;�ݴ`01�~�Ù�Rݷ�g�\�<u��B��:S�ӧ�Q�2{J�?��S�Ή:"�vG��&�)����;���ׂ��u��;�R�[�ҙR���x��	u����[��k�2.G.̫�5F��p�B6�����x�"	9�\8��k���B-\��3{���ˏ�
տ"`w�8w��3CFO�#�:5k<��jF��˨ḩf?]�=��=i�q�,��#g]/F�\t_6�:��0$<�Pd��N����(����jW�
��!ӛ]��٩�>u���"�c乻D�CJ�
�às���A�v�����s�A�A�����"�?峵� ?/�P;40�$l�B#��� Q���o/�t�_Sc� �|Z��H��jw��@p {%DXĹOj ;]��[�q�/��:�bH!�O˻��s�j��ig�B�d�m�t�WV�c,oу7�G7=�Y�����9r�
�������v	��u�@����!�������]z˛���h��3�!�Y�6��~�zߧ�b����@��N��4E�>{�2�`zȷ���l�Y���MUvqꡗ�~0
͜������G9�:Z�x��<1��6kjj��*B5�>�(��2����A��Ob�ǭ�ө��>D���ș��d�&�6w�����4�]	�Šc��ʱ�h�z�~���`���:�B:oW�A§w�F^�������}���=t{YѥQ���W���n����+�Qm_�E�R���!J���U*�w�7=��-(��*�������I��U�g�b8Fh9��j�us�i�'�`p�u� ����\��Q���|�.״��_{"�4��Ȫm4��k��o�V1�-��8�G(~oH?�\����n}�L\�2cѤ@��1$�3~�������
��j
���H�*q�	�`'*���hx��d��3�^m�C�[g���K�!�hA����ֺ ��@0��&[|��� 5�W�tj@z�@�C��/�M�?y޸	�-���K������oI=	��d���-��R����� QЏ�>Bj���NIF�p��z�S�shV��K�pƈއ��o�	���D���o+�Yl��L,+7H�o���'_|��(���d���#���⓸�FEN+"�$u~ؿ�=dKw�g�.k��ΔT�������/kՐ텎h]��l�@a���.o�Q��_��_9�x�P�F����������Oe3z������Ou(� ?f���(�q�d�>ԙ�C#����c�/��ŧ�Q��Sr��c�x�׏C�o�Mk�;Io�>|�O5x�B���/���#ѡÅ0���R�~c5$���� ���/X�iA�����y �&��(����yB8�'��w�y��E� j� �1�3�y� @9;6t̳E�)`��n�wP�ɀ� a;�P��D �bb���ďKh���>^�
�S�+���r�t�ND:�:~�(�9M�����*���s�|G';9����~��,�T��&f5q�\���_�%ο?itڦ�!^���\n�(9���X�q0j���E�2�?�|�J���~������b/k����|b�Z��7�+�"���3#i�����=�V�Y��
u������{�׻w�^��pi��M�Tq�u�jѳG?W?D�_tyȘb.��P٫Ŵ��̛��)�=C� w�����f��|�)]ڷ���"!�Us
���p'og����;��e��������I���d���x�+�ۛ>Kj��l\�s)_�a٦Ӳ�����[����H�]����sH����`�����L����-r�����R��5!j��	�z��6�T'��>�t�N�g�2�Yي.'�4��v~*��%��r�؂H�''3K�~��lD�����l�T�d��+��.9�J�u��V�ck�Zڻ4%#&�5vnZ����D	=��D���;:}�k�����5�g9x
�/SK��p���I�%Uߝ=���%���r(�ׁ���'�*&26q��􁐷
&}�y�;�ԜzL�*�T��1.�'���,q���K��1���,�0i2�*;I3���5�0�i��Lt^ڹng�~�_^�K�����2�W)��wv��)ξu�7И>��!,}�AX��~�����Xh�嵅�Q��
��\g�����!-t<�ճ7G�X�X��#��p��+�C}즅�K+��9M�|z1&�@Z�	��I_�#�[fP�@����Jo$��H�r��9�_�U@��
;��DV����9QRP��E��+��P��^2�"�O$}���<�Ґ�����@~���{��lJ@���NA?�+G2&B?VL����[��������'��O�najw��泏�o^j�Z���v�X�`V����ag����3Z5�'�B;O�����6���yy��͑Oj\OԈwn�ӷ8ɾr���ܬ��M��h�F����Zr_��;�p�Y��{�s~H1��lb���ԑ����L�c}��*�T��.�q�h���߷	���U�<��G?$ya��{��b^M�8�I��c2�_M�۝����6���ų�]�(�����v=}%��l5��|np΋�Jc�4Cj�����N%,iǄ�-�8W)������*ݟ�t裺�sm���}�D��z��)�D��ŝU^e�&]�N���^���le}�MM��9���&������Fꌼ%�5ߍe�}h=htS�jj*e���'1�v����L���F&=�W�C:�7�z{�)�>��B��wJ���fR�_�$�^T0wk��ܶ��XU'����a�������پ\'�u\a;��D�R�c岵��l/��[�d��o�P���x�iӱ�kZ�7F{Q�����?; ���r~�����H�k������~!Mn n(>xkG�YH7h%�k�u��&vЎQ�㚡k�I��Qዐ�[<o�{X_���Ca頿a���pBr���d����'5��,�����7i�_�^Я�&P��8�� l_�%^�ԏ�WGxy8�K��s�.�]^�V��G�foz�r�����x}q=5�yA�������7��Z9�͊УY���Z:�'&�K����Cޜ�V<Ö�}iS���w�l+H��ְfs���vvZs2�ݥ�6}N'U"�H��֬�MH��O*Y"������꒽��n���.-e����>x�	铟�}}�GΣ�(F.1���(�~.�2��8"���5��b�xe��]����W���G���6���lM��7+ߒ&�P;��<E�'Ɍ�^
	�"J"�)�[AS2 Ԗ����2��W0���&P�~���:ҙ����
3���l�]5���@�|��& ��$��
`a�����t����o�o\���D��I�xM���߳ ��-x.\'8D0�v��}���ּ�#�v�÷���oc��K����7{y�gm�m�2�zZ�����Z�x��`�q���u՜��6\f�?���?�<��~ tY��tm�o�w�S�c��Գ+d�/
�c���gx"`z!��,�%�%���^�s�Ѣ��b�l����sC�>�G}�~��+��?߭�&z0�S)㬖1�u��_S��B�
7.����bs8��	�/���S|
��3��l=�ZO�؞���s*t?�P�,�s�7Z�,�7T�����
g�G������x��l(zL���_-�Q�ΉC	��3q�
�#�uwW?bη\p�.��O����S�4;=�=R;a�A�n�-Q*|˚f�M!���~p7��Iw���Ճo��=s��O`8���d��Wm��S���`��P�̍�8|���]vֆ5�Oؽ�����o:�78w��G��b�ĭ�������7����QYzR�t�Zm�D�׺+>�jd�Ԣ�x��?�ևxZa�p͎����B��Z.+�����O9�]
������~����%<��(�j5�?v� ����4]%�x�<T���׽�˝��"y�2~��ҋ�i���_��9����5�Y3q�b�N����3C��o��1_Sq�ſ?^v,��h��j�b^�qm�"���*�}{��̜�(���t�]��s�&F]���R9�)�u)Cb�p�LD���x�\&Ce��n��S���"�'c��/�x,.!y�V�w�e{���Lw��J@)�|�����cl�Q��D��v��]���9=|�-���������Y��7%�J�T�Ͻ���W|~�렗Hƍa�_t��ir
*�h�Q����.:��੶���h���fl��ՃR|̗�5�3������w�}Ƽ��������Tc+����*X��H���_��K�xp0!������˯�Y>�V���c��e]ر��GA�Ze�]ܨG�����>$c��$^n��Ý?�������_�m�8�A�&�ջ��r��?���z=>x&B���Ӕ�ݏ��IYG�����GE��''2���Wo�*�3It.�8�Q"_U��5���@���@y�y��j/z�7����[�R�|��-[��o��Gfw�'�u�ᔴr�q�|��d�´k���{��o)w���[��2Y����S��l�wh�|��Փ��]�:�~����g�I�.��0~�a���Wa��U�����7o%��ERCmD��u8��iw�I����G�ᣬC����h��6��D��L�w�.�|g���.z��v='�S�o�Pt�uf�ث�0���s�W���_��P'�KK��I1j�=�*t���}.���)�-���٤�~w}}YBct"7Bs�!�:�+���}�Prg{N������W�j�N�q\e>�;زZ��Դϔ���_�
J�y;�~�_/�r��{�d�Ny���:VB�!|/�S����7:�7�n�a���ɤ5��A>�ٕd&%�����!ϥ��[�b�@%U��O9�E�X�W��Hu0{s)��,aw�z&Ǉo�jGh���ى��UF}�s�Kخ���q�.�$���>s8�=��.C��A?�ޛ6ќ��đ-�5|Dd�Lϩ�r��;�-�Ͳ����G�s�� ���>0/��>��e�4��x�I� '��)R7�zf��U���r���$�{�:-�#j���Lj1�e�����a_L�h��h������%9b?5m�7+�%�~o"�t,���N~��>v��������c_�.��$�Dlw�bZ�c)F7�uə��v�=I�L�-n�~Ͷ���Ϋ�R�"��Q��lU�)?7���Vg�_�6�dz6�}x�QW[À�ێOZ]O�|�]p��qƼ�"5/� O>�N��P��g�j�^49!q��U�J;�w�n���9���	�C���y5�;��&��Hˍ"O��M��6�	��ޖ~�?��v	��瀉w��������}�B��R�(���� V����&Z� �| Py[�7�� �yj�P>��C�g��+@<0擺�A@?�(�<}��6"#��	j��<q�������U�+��[W) �B�tv���<���������l@lS���q�7���vݏQ�Xu�>"dV���c��w�7�����͐Ϟj�C�W�e_3*���������'�����&oD1��Lh�bW�fm��s!�ȋ�WL��Hn�)�M��M���-��&�i���oGO��R�y{w��fc簨�c�Oa�K;�W�ux}#[�����M|�^O:Ƀ}���P���Q�ޥ�e|�A>�{o�x��D噔�j�ӟ>�����<�~풢��D�kϳ�����j���$OSu10�s?�{�/P�J	!���Z���M� ��{��u?F
�=*dH��1���>�[JA�K-Dߖ�n���;�@<hp8"\*�T�h������P�\/�8,���h�̋�Y� ��?�s$c��h +g��!NE�ѧ<Ч��@řcx�Q�հ�e����)y*dDU�����c�dt+/�d���qn�8���4;94���h��km�mo�o��p�~��������0~�����"��	�?:O�!**jj*J�����qr����#����������[\^����ߠ��@��fFp����s3ą�������I��,qr��8����������}h��\]Z����+l��c�1\O[GkYAY}erb���kA~TPN���������������~brniy�:7=J���1��loK��izCgWokUA#��"�y��p��AU�	б�<��N)���C����"&��8ږA��n�gd�oP"
�矃��Ω�72+�i���a,���y��dv*����=`Q%�>���Ѝ��Ȉ�)鎕�*d4ή �ZW�G	@�W��z��H���������������������}<=<�|||�Q��(/o�����~�������� ���w��7,�7�_�H����<<A����ڗܮ���	=A9/�g@`` �����@P���v���~[������U����������r���������/tr"��������������;99;������wgG[:;��U���r���ݟ��Z��q"�[���9ؓ���ڒ��������y+��.��z�n���\?�=P��҃�#s�3�K+�<�����ցik�l���q���������� ��������;<���II�$y�2;5�4G�w�W�EV4�w�t��O��N/.��L�ήQP"6�����x������,�$��V�IZdtj|���[������V�88��\W�P]�dd���O�C�L[KcivqCunJ��c�}Z��GZ9ft|rv�8=1�����0>8>������8=B�`z{0<v��5� �����,3���$�M�8 O�� �;����/ hY
��oy�0Ԗ��" 9mYZ�u�s!����)Y ����,��{�l�',@=��@�'����h���A����:,���R��l��ڴ�����2`�`���RQ�BƂ�!�<������@��N�����[\���B{ݳe���t�Wv��=N ��TQ	�[&��_��$��@���b^_�ƅ�^Ӝ:2�N�ߦ���rv��|P�]��nk2f���Ư�2ЮyQl?%G��~U�`]�M��	��C�%c|2� ي���!�X_�!��CCLwWKk���$I�-,��,͓���AlGK[[SSvvAIMMCCCKCM�U)(��3����E-�C��٥����u�bsyAIEMC�����ّ��������&���r�8���IAI1�V��]Ն�������*2��R����0Ӌk�sャCCئ���k��:Dq �J\\����[�[ޠ��\�^]Y���@��\_���F'���1lGG/Ӕ�3��7:��]��O��.!;+'/�� 7+�����(?'����W��W\RRJ���eg��1�����8?'���_\���7��n�+.*��r�MO.���?�䒨����H-������Ҳ���<��/��zrH�����]o1	�C_��q�7��C`@PHhXXp@PxDDdxhPHD$�II������	�A �ȟ?F��H�$FBr:�ȟ�S��V���@����a!���LO*O�	�C���П?��"����0�	���#��U����BC��G����/�D� ����������S/�G�������������u�]�s���&燑���a�yI�FE�����w?I[������$Y�^�:G��~T+�GK��l�
�dN��wm?�b�X0ir�eЯv?��T�+"�� D�=o_
�ca�cc-�Ⱦ>`�$���X4;	�b�=m ݣ��B[[���<�w"cx9�g�a?
J����XaN�8B��=jKTȘ>:�@�I;��U��&Na:��;���[�j���ᡑ��)�>gr|t�]SP\���34ER���5I+��,.��LtU�X唷�����Ɖ�$-�J�[Y� �R"V'�G0]CĹ������啵�����
b}��a�\��=��8����ZY�֐t棱_^�������ahp�����������yX��dn���������$�`sf��$�d��X��%��c���е����ʺ��������<�������>+!c6�c��Yy�d$����K �z�/�����#�#�R��������@=F֟9�Pe��	�������n/��Gno�����<���`��O��� }�!ߡj���:l�.���h�-�۲�;S�	*d�-�|�L��L��-��'��A��YV�u��Ѷ��g�ޜ�)���/�g'ACD��C��������c� ?��i.���S����* �!~�稊B�G��@�-//�,JJJ��~ ����{��:	�cc��sk���s�s3���~,;I*AEC��������9B_k����v���������\���[$Y����)��:3:>955��2O�OV����ǈđ����䆺��������ʁ����4���iՅe���XRgF�u�٩yu�Q��v�_����GF'g�gfI����q`dbf�exfqfj������ל�Q����R^�;@Z�.�0��c =��Gy"@d�-@�J诽��t�o�>��l[�*��Ol���'E��y�yI��O��d[�˒1�_�6:`N�ҙ1�hEC]`���>.���҆�>-���s�`?��U�Z�I�G�=��YU���Iʡq��$ً?7=<��mmi� Mv{se#if���)��$�T���uc��VI{�~��bs�$�G��m�����=��(�SȾ�������uҾ�27985���� .���|n���6;4��^�#��X�W���u���w`���j�#Kߌ���8qf���b�+���Ё_��}rF�Ut�O�͒����&�6K��]APPP�-Ǉq$&�c����ӛ�1�5eE%�#C�1�%L/��,��ml����z{�������,�0��N,�,�ϐ6rC�V�~kl��a&W7)�H{Pj�<N�f��&�5�7����fjddvqqnvjxt~�4�ӃC�#�S��&UH�-�����8~��(;b+ʫ:[[�ۚ+���]��.ʏ��/�-i�����O���˳�
j�s�Ãl�y�����ݣ�c�s�D҄�-/̎�F����gI+�4��������b�Ԫ�ގ������������#������L��Z[�9:8���=�?�g���:9tv����s���o�m���_�廝�����3	�IԎ�VV�v�6 滸�:�ِ��O$�cm��~����ʁ��#ɗ����8�ޝ�ʲ�-�"�?\��\�L��ǯ�] ��ږB?���ex4�O3���9���S�h!����ь��P2��Nj	q�/�d(�����k�-��ۀ�&�w�-,�_]�a�8�Si�
��&����n�@IMEECK
4�KD�ƚ$Ό�Z{�VVW6ȧ����3��C��驩�Jj��@Kv�Q���1lU`I7?87�L�&d�Y�-�meZZj�AOE�����AA*���� ��-#K��kK���V���in'�7�$���H,l�onh�ŏN�/�.͎������F%ǸE�!݊:G�IV����:bsevhf~�kb��<;=6>=395>1=:��*Jo��oW;5�gtl\<:&*:1))91.:bLlbr29�NLLL�����O atlL����M"�'�c�HQ2���W���r[t�S>҃��v�11	���ǡ�$��I�ظ䔔���Xr;dD��)�1�_{�[���<���?�������~�n�����2��4�a�e8� to��c���<ֻ����ؗA�f�	���2�G�q&�v���<4��n�l�=�kNƔ�Q �Jg���R"</C��l7� &b�����ܪ�s#3J�\�.�0ib<�[E)h?{r*q��-؁���ѩ�i�^r �VQZS_���\�XЀ���
����q��啍��i�5M��+dobcu�?�4�d�$d��uuՐv�<�0<�0;=18D\Z�\_���a��^���8E$������D,vz�82��+$<<������'5��Lj*��������������alW{{[MZayIRrJ��T��Oz�����������������������i�2�����ڃ�`H�����ך��у%M|V7�Ǉ`�������ww>6.��ѱ�^���y[��m�޿�KǲJPκ�����D��h��ld+�3χ�[�����	�2f4������-;���BƈA,X���V��}��}
Ϧ����"�<aM`d�`��o]X��=M$	�ٕ�����M�����0����������Q~����ڷ��6�+�ؕ?�?c�%y�O��2�J�5U�*��AH2��8�x��`c#�X�f�Ìm06��.�$I-	I-�r��>��㚪�|T��=绷��r��V#r�t:y>x֗���p�X9G�3�G2>%�sѽ��n�ǵ��H� �ͤɂP*��D���g-�A,UHEc��E�m6_8.���:Bs"��ҍ�ť�ݝݝ��Սè��쿞Ό߿�;7���D��L<���8:6��f�����O��y�����h2�)�d�Ϣ)��X:w�*�Ԓ�`� 	���_s_�:���o.-��-�)�O�q�fkk[}o���4o�e�y-·��;��\mw�Q>|��4��pg�m��������T�I:��W��Ƽڊ�pķ����7o�F|+�c�m��f�l#l�T�����Վ��x����k�����"���z0Ι�����9��9:Cу�ooŻK��Ŋ(������wu����
N�0P�&Ke!���5��Y�:�b!�ʔ+�r)��s��TK6S(���Q6_�J1{\D��1��kl~�����뛛h�{�#���������WO�^��x¨f�8�,��w��?st�}���gݏ�|��Δ���L�R9��i>���8!�:���8Z�������}�������D����Yr�GG\N������5�R��I�C#�c��CCã���'&'�^"�ē��jv\�52>.�W����ވ�M�q�F�F���q�#n��>12잜���<%��)܎k�~�c�<I{�va���Ϟ���.΃A^K��g(q��-���ˉ�C���^9)���r��N^�������zT�E?⪾�9j��r<�W��g}��㽨�ڮ�٬0��D0��'���.v�^�(�D�p׷J��E�\��2|2�������� ]¿���
�(�W�ٚ�fdɻ�H���8�FE�|��ߡ�_�s�r.�KbE�.��VIE)s+_M/yP8�Dd2��]�������Ã3K~��,���h`cmv�����'=���y;1�+�W)
�E��P�t��\2N�4(�����6�q׻�����a?�f��~0P���}\q��=�,��o��"K5�vX���u��y�2}���`��k�����pS���t�s����ݳK����u��|X��7�OLE��b��2G�9�b�{�4/�_k�<������~$q�빟/c~2�q}f�����G������{��vG��j�k�l�t9�]6K����-VGww*[�]]]w�ֻw1[�{zz6����N[���p����!�ר������N�v��6;:����C����{���V�=r�Gڱ�Z�<m�x�ݺ��0g��Lp��׳�F�Kڧ�Wi\kE�ӕH7q�f����#������;N�O�����$��T��d��^��^{E�o��S�_��D��<�wCw�G��t����G��[���67�����^���z���������)2�\ar�=o��y(����m��<��_�g8��L�?��p����c�\��9Q��ޮϷ����|܌��|6�L}�t�Z�y�ӏ��_j��U����9N,������}���ţGi>�H�%tie�/�?�+��l2���䎅B�(J"ǉ�oE�,�kq�[���~ �w=�c��������3�	%3��A ��+K�+���o�_���������,/��+d�%ԕb:-�I_s�<�%c�g�ΗnO��������J���Τ��n�|�U���V�C=�Cf頝�F�*�?O���C	zuJP;A5:*4Ƞ؏RY/��I_t�t0Kϲ���I���� ���X<����N�d�Q���A;,C���,�
�����<��z�#��k���
�YlT�����(�M�1�觠��nj��=�ZR8b�NK�K��)������}�q����,��s� ���8�f�3�n����k�K�h'ܠW�5�Y�ұ�*�?��~S�]��ԩ��A:��/\��o���I�7s�:ɮ�F�*�?uL��$�UK��7��O�R?B;aN����b��eWa�yn�AO�!�Q?$�ǘ���Ȁ���� ��f\��Bbh�z����,f�Yv����R|b�EOC�*�_q���ud���,['ei�@;aN['ٵب^�a�3�7��6�@q�He�̪'}ѥ��,=�n�1�a�z��Ю�,=ˮ�p��|�'�t?��D}�TKz���(cR�eLJvƟ�/� ߁�C��7E�ê%�~�ڡ�!�t��b�z��� �9� /l�+�����Q�)S�~6�h�uf,ԯ��)������}�AVk-)��x���f�/��p�^U��g1Kǲ�0\�g8�P�oF�7s:u~�t,�
�|�, T�p��Z�V�(�X��U�m����Z��ܿ�1F�	:�u@?u0UuS�O78y��%�#6����`<�ʐY~/���3ʘ��'`�i�y��f��y�ʘ��F���(cR��0\�f���>se&_T����,����Y �AW?�2d�e-;�Yz��(c��Z����]�Yz�]����0Ν�t�Lx�>�2d���g�2&%�	��N7�1��V�Q��Ɖ�-�]���+ �y��z��-֪���Z�ҳ�Z�Ie��j�0׬���ب^�1}6�4�*L��0K'�Y~�
�uPTREE  ����������������                       .�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^c`�`��Q0 �#  xE�TREE  ����������������                       y�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^c`�`��Q0��
�G(  �%�TREE  �����������������                       �             N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OCHK            �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   ���       �             v             r��sOHDR�         �       �           ?      @ 4 4�              �            �8             shuffle            deflate            �     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   �u?�OCHK            �P       DIMENSION_LIST                                    u�        �q�&OCHK    T           P       DIMENSION_LIST                                    u�        �)W            ���OHDR�         �       �           ?      @ 4 4�              �            �8             shuffle            deflate            �     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   x9�OHDR          �       �           ?      @ 4 4�      7     t                   �8             shuffle            deflate            19     �           ������������������������    �     �                       \/�MpOHDR/                           ?      @ 4 4�      ��     �                   �8             shuffle            deflate            b                 ������������������������P        _FillValue       ?      @ 4 4�                               ����             x^���c�a�9| �e�"5��<X���aĀUT��
x�
�}>
,Y�h���s�Λ?q��)S�M�1sV]}CcSsKk[{GgWwOo_����Ȩ�ظ��Ĥ�Դ�̬�ܼ��¢�Ҳ�ʪ�ZtӇ/  �S��TREE  ����������������L                      �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^UO=��@��eRو`�H���4"i-��sV�.uJ���m��E�ofwfyy����!��P�#���=>O�2fT-O�49�`G�I��~픮��f{���BƼ���j��V�f"''
vTluq}�Z�7�C����m!��Ƭ�j�ץ4�DN�(�Q��9D��;�9��l���Z��$1�h����ȩ;*:�h�����l��9X�X51�C5��'M3��]���"�L&J�}6�{|62�Y̰hv����� vT��0B?�S��0L������i����i��|HE�������UU���O���s3��\�����������TREE  �����������������                      �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^m����@D��"l�@��Z�Ac6�H����y��c?ҿaZ����qF{�Jl�=�R&� �,d�v@@ʪ���]��̀�!�Ǡg9!��R�v����:���Ė"p�H���(J"E# A�����Q��̀� ��<=�	�>`��:�d�:OP>i�*omi�D�k��vҔZ�iȇ���>��-�m�cг��V
i�hs��ս���Ֆz�J����_{�5`E����Z��̀���<=�	�>`�����:OP�(�*.l���D+m��Q
�&�ӻ��>���(�cг��V
)�(b����W\�������W\~_q�|ť�B�|�تK����W|���9�V�f�\N��<i��,�r.3`CZ>�����}�P7඙Ǡg9!��҇c���ޯ���f[�Xރ���_��_?������}�K��oa����m~JTREE  ����������������m                       �             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^���0�Ҩ#�0�t@
���]�m*$���d��ӡQi,<��u-�Aق�Ѝ�4��׼�o�F�h!	_�Q�-�݈�!�RH�Wf�A���	`��\�	�cTREE  ����������������                       �0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�         �       �           ?      @ 4 4�              �            �8             shuffle            deflate            �0     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   �-�OHDR          �       �           ?      @ 4 4�      ��     �                   �8             shuffle            deflate            �J     �           ������������������������    y7     ^                       :��}             n             M�OHDR�         �       �           ?      @ 4 4�              �            �8             shuffle            deflate            B     �           ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                P       DIMENSION_LIST                                    ��     �   �_OCHK    �     �      ( 
   t  REFERENCE_LIST       dataset                                       dimension                                                                      �             v             ��             ��             �(             n             �+             q*             ���OHDR�         N       N                  	      �      �8             shuffle            deflate            U\     N           ������������������������H     
   is_result              @                                $       dtype                boolP       DIMENSION_LIST                                    ��     �   �Y�                                         x^c`\�P�  �"iTREE  ����������������]                       �8             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^틱 !ٌ�X���c����+������ӟZs1s���L��:P�3��JgL�Aq���-��2�S��@�,+]D�1���Ѻ�TREE  �����������������                       aA             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^�NQ�1��%�Pʒx���z�|�u:��w��MK�1j����8�s�xkI�Xa�ls)���Tշ��2�$Y+
+^�X++c~���Q"Ao�`���{8\�=����uMި҃Mw'�̓�#��L����TF���f
Gs] ���:/Rw?O,�|�������.=]Ç�&%{qTREE  ����������������T                       OJ             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^�9  ��O�˖��0Z[�� 	�M��d%�K�ADb�Pr� �0>�� F�U��Dd�7�H�b�4�;.�x@�[�5360t��TREE  �����������������                      �Z             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OCHK    f�     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ��             G�             �.             *�FHDB jk        ����       force_resource�.     �       energy_cap_per_storage_cap_max�U     �       storage_initial�l     �       cost_energy_capXn     �       "cost_om_annual_investment_fractionW     �       cost_om_con�p     �       cost_om_prod�r     �       cost_depreciation_rate�     �       cost_storage_cap��     �       distancea�     �       lookup_remotes��     �       loc_coordinates��     �       inheritanceo�     �       names%        OHDR                            ?      @ 4 4�      �X     t                   �8             shuffle            deflate            �d                ������������������������    �     �                       ��#�BTLF �        <  ! �        ]  " �           �        �   �        �   �        �  ! �          ! �        6  1 �        g  " �        �  " �        �  5 �        �   �        �   �          ) �        F  # �        i   �        �  ! �        �  " �        �   �        �   �        �   �          , �        A  & �        g  # �        �  ' 1䊠                                                                                                                               OCHK            �P       DIMENSION_LIST                                    ��     �   5���OCHK    ��     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �1             �4             |             �U             �l             �ī                                                                                                    x^MM=��@���#��7�D;Ks����#�@�6)r��D,r�v",����o�y;;�޼<���R�w�YC�3�_Ϸ
�>��D�/p9�xj��mr>+Ž1�5�����6x�U�mG�&zx����N�<Ƨ�C)�ɬ���^���B�<b���{h��t{<*�=7�5����|:x�U�tO�&zx����^�<ֳ�O)�ɬ���.���B��4���\�>��i�����*���DĹ|ٔ��f��tBGC2�A�G�B�Hz��T��)K����hS�)�$B�X�,�i�bѲ�R�Z*qS����gbN'4$I��-x�?����sʊ��o|gm۲�ʪĩ%0�E&[m��ĔOZ˕xØ�dj32�����?� �TREE  ����������������"                       �d             N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              x^cdddd`�
��)�0�\ � 7TREE  ����������������"                       �t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDRu                                 @             �            �8             shuffle            deflate            �t                ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    ��     �   ;;SmOHDRM(                j              j           ?      @ 4 4�              �            �8             shuffle            deflate            )}        j           ������������������������P        _Netcdf4Coordinates                                        
   P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    ��     �      ��     �   ��OHDR;(                f              f           ?      @ 4 4�      �s     �                   �8             shuffle            deflate            }�        f           ������������������������H    
   is_result              @                                VM��   ZxgOCHK            �`       DIMENSION_LIST                                    ��     �      ��     �   }q�OHDRC(                f              f           ?      @ 4 4�      g�     (                  �8             shuffle            deflate            )�        f           ������������������������P        _Netcdf4Coordinates                                           �~j5           TOCHK    u�     �       P        _Netcdf4Coordinates                                           P       _FillValue       ?      @ 4 4�                               �f�(J                                            x^c`\�?~\ a8	�������AN�) {�ITREE  ����������������                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           TREE  �����������������                              a�                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^m����0�'��`S�K����l��)�#A@B�`3��`p`���,�7��=�Iw�?3��1�k\^�����N���E0�!�p!.�p	Ő����H�bA��.��L��u[d����KYk��n�g�"eȀ.421���U�Rd��`�nyMR2A�V��mPPO<o�������Fd@����󦞊��Խt�o,P���	�������?�^꼗�����:t"u��x�x�bHRg� ��,��wBv�����]Y�+���K��X�V�U�J�
2`�Wb���W�PI�U:��o��VBV�YW��gG�'Oc*�bٜ	M����)�3�1�~<��t���&u9&�	�J�G֎�a6��,c9m�;�|��ax��t7-�9��,"-�v8c���lx?onv��f�p�z�CsL����X;�����ir�/��?���V�������/�#���K@�TREE  ����������������                              u�                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OCHK            �`       DIMENSION_LIST                                    u�           u�        ��|�OCHK            �P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    u�           u�        [�M  ��
            Xn             W             �p             ���]BTLF �[�� <  ! ���� �  ! �_}� �   ��� �  < ��� d   9 q|J�   , �F.� @
   ����   # Ѧ�     )�:� A  & yI� �  ! Da�� g  # ]3�� d  7 �y� .  ! �X� �  , d�� S   `��� F  # �t�� �   F�f� E    �}"� �   ����   3 7�� �  $ ��� �  G d�� �  " v� �   ���� �   �f]�                                                                                                                                                                                                                        OCHK            �P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    u�           u�        G��OCHK    ��     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      P%            Xn            W            �            ��            �m��OCHKP        _FillValue       ?      @ 4 4�                               �43G�S�OHDRC(                j              j           ?      @ 4 4�      ��     (                  �8             shuffle            deflate            r�        j           ������������������������P        _Netcdf4Coordinates                                        
   2*�>                         GCOL                        �                   s                   �                   �                   s                   �                   �                   r     	              �     
              �                   r                   ��                                  ��                                                                                                                                                                                                                                                                                    !               "               #               $               %               &               '               (               )               *               +              NL33::hvac_transmission:NL32    ,              NL23::hvac_transmission:NL32    -              NL41::hvac_transmission:NL22    .              NL41::hvac_transmission:NL33    /              NL32::hvac_transmission:NL33    0              NL31::hvac_transmission:NL32    1              NL33::hvac_transmission:NL41    2              NL11::hvac_transmission:NL13    3              NL11::hvac_transmission:NL12    4              NL22::hvac_transmission:NL41    5              NL33::hvac_transmission:NL31    6              NL23::hvac_transmission:NL12    7              NL23::hvac_transmission:NL21    8              NL32::hvac_transmission:NL31    9              NL31::hvac_transmission:NL33    :              NL41::hvac_transmission:NL34    ;              NL13::hvac_transmission:NL11    <              NL42::hvac_transmission:NL41    =              NL13::hvac_transmission:NL21    >              NL21::hvac_transmission:NL22    ?              NL34::hvac_transmission:NL41    @              NL12::hvac_transmission:NL23    A              NL41::hvac_transmission:NL42    B              NL22::hvac_transmission:NL21    C              NL12::hvac_transmission:NL11    D              NL32::hvac_transmission:NL23    E              NL21::hvac_transmission:NL23    F              NL21::hvac_transmission:NL13    G              vz     H              vz     I              �     J               K              ��     L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j              #9958D6 k              #FF009E l              #030303 m              #FFE400 n              #FFB600 o              #275FB0 p              #27B09F q               r               s               t               u               v               w              #D800FF x              #CF74B2 y               z              #6F6C7C {              #072486 |              #AB2C18 }              #33F6FF ~                             ��     �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �               �              storage �              supply  �              supply  �              supply  �              supply  �              supply  �              supply  �               �               �               �               �               �               �              demand  �              transmission    �               �              demand  �              demand  �              supply  �               x^m��ʳ0��v���Co"|���x�q�8�f���(Y���Y�	�m;�g0o��N(V��eO�abuu�t���]<P�$��Utu����S�W�Q&��;@'r�+j{�zC5h*7�2Έ��^B��tI��R�g *��%�F/�	���l+ep`"�L 
�$4�j3Q�e(�zӒ��w����n�D�u�(��@�'��u�7C8�������� ;/���o�=7��Nv��]��؆l���=��lۆ�@�%{a�{h����fi5p`"s �e
���&�`���4�:�uMZa����V�I�qT��FM��TNU+�5�u>*���D��Tړ�����V����[�f9�2���i~;+�l'��ir��&���s3�]��NiN��Q�}�6u�NZr�����R[;�#�}����`x��c5���Y����d��'�j.1���&�p`"� ��H����h,��8��������x��~������?�������o�c����r\TREE  ����������������t                               ��                    f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^��� C��1�n�f����@�b�� *�^�Yk�`!�6�ԽWn�YJ�`!�6�Խ�_��w�`:Ff�$�m�Y�����m�)@t�DD#H���Yg�%�6#��?�Ҩ�TREE  ����������������                              a�                    f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      OHDR}                           ?      @ 4 4�      ��     t                   �8             shuffle            deflate            ��                ������������������������P        _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                �gM. Y �OHDRM(                j              j           ?      @ 4 4�              �            �8             shuffle            deflate            ��        j           ������������������������P        _Netcdf4Coordinates                                        
   P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    u�     
      u�        ��OCHK    ��            x    t  REFERENCE_LIST       dataset                                       dimension                                                                      y             �"             P%             c�             �T             (�
            ��
            Xn             W             �p             �r             �             ��             d�;OCHK    ��             �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ��             n���OHDR�                                                 u�                  u�        8             shuffle            deflate            ��                ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    u�        ���w    x^U���� D7�:�Ӏ#� uD�t�N�9p�YI����[�E@�����ZS��N�*�V��9���l���+��N�*�V��9*��M`6�B
.�)[�c�ª�9G%��I��gJI�I8e�t�_X�>�(�6��|u�)�[	�l������U�æ2�{�U�Q�p��>�IU��L7~���稌2��q˚;��s����;��6��v��w����{f����ɔ����!��D���2��v-����"�Ez��T|�TREE  �����������������                              ��                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^��-ά0�ÈCXnT��]��k�%���f0$HhFTU~]�QG�a�5}���c_ԅ�mS3�/��֬��a���� ��~2V4`V��m
[x�Xt�[� ��(���y8�1kK~�$$�P���+I(��d������W�������@�^�:h�	��CH�Y"{8�IB"�x�7�� w�x{9��Mā����7@�y�������K��bt�� �������p�sBlB��J+�*����*=*��#mۣ��M}�9�ʃ���(_��=�BM�R�!$*���:Z8�JUJ'gu:�IDqg��:��>�'=o�:�$zP�%`чN:ݝ4��	֚BR!��f5=�0G��4߳�*������m�eJ����2�� �]��6����2o�������A̹�4B��o�r��a�p^�i]�e��#n��qZF�u\�e\n�Ղ�8�+x���y��q��A<M#$
+�'�ӊ�q�����?���TREE  �����������������                               ��                    j                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      x^]L��0S�!2�kϑ:{x /�"
P��� 	��鮽�u��%Z�7#�~��e(т���ݨ,C��͈�O�FeJ�`oFt|C7*�P�{3�;`���7���V�9�rl���-N���N?z��;&pet�#,��M����\��wN�N���������E���wTREE  �����������������                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OCHK     �     �      �     t  REFERENCE_LIST       dataset                                       dimension                                                                      a�             ��             [�OHDRM(                                         ?      @ 4 4�              �            �8             shuffle            deflate            ��                   ������������������������P        _Netcdf4Coordinates                                        /   P       _FillValue       ?      @ 4 4�                               �H    
   is_result              @                                `       DIMENSION_LIST                                    u�     H      u�     I   ��OCHK    ��     �       �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ��            �05OHDR�                                                 u�     J             u�     J   8             shuffle            deflate            �                ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    u�     K   ,��FHIB jk         �     �     �     �     �     �     ��
     �S     �     ��������������������������������������������������������NTو                                             ��            (�
            ��             �O�=OHDR7                                                 u�     ~             u�     ~   8             shuffle            deflate            �     w            ������������������������H     
   is_result              @                                   X,� x^;1Ɯ��c�(5�?��`N_����}[>��fY�]_�����X�K^)Փܣ�-���5{6OYyc�I)=��'7Ki���a�dz��Go�A'�4��}�������O��𿍇�y.�.��F��ߛ\���>�i���b���|�.��w��8Nc��bx}Z����î5����dq��	���	
��w��tt� IVb�TREE  ����������������;                       ��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           x^���h J��x �n�@7WgG'K+k[;{m]=}C#cS3st��  Ԓ(�TREE  �����������������                               .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          x^{|y��N��1޳���ڿ���y��%�����_��~���д,&8z���[RkL�$o_�]�����ǹ�=�S�:�T�����+�
�v�:�x?O���We�|M�h,8r�/�k����0�ᵕ�7᧏��ɓ�L~�m;³�S׼�G�������	�@�S\BLXD\ZHRPJ�  ��W�TREE  ����������������=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  OHDR                            ?      @ 4 4�      4�
     �                   �8             shuffle            deflate            @8                ������������������������    ��     ^                       {��8�OHDR�                                                 �                  �        8             shuffle            deflate            �/                ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    �        �B#qOCHK    �     �          t  REFERENCE_LIST       dataset                                       dimension                                                                      ��            (�
            ��             o�             %                          �,�>FHDB jk        R�B:�       energy_cap_max_systemwide     �       lookup_loc_carriersp@     �       lookup_loc_techs�B     �       max_demand_timesteps�D                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               x^cg$ @����q>\ �h��ښ����¢�ҲԴ�̬�ܼ�t�  	/'STREE  ����������������G                       �/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           GCOL                        storage                              ��                                                                                	               
                                                                                                                                                                                                                                                                                                                                                          !               "              Hydrogen-to-power storage       #              Electricity import      $       	       Lost load       %              Photovoltaic power farm-scale   &              Photovoltaic power distributed  '              Off-shore wind power    (              On-shore wind power     )               *               +               ,               -               .               /              Electricity export      0       $       Transmission intra-zonal electricity    1               2              Electricity curtailment 3              Electricity demand      4              Combined cycle gas turbine      5              Battery storage 6              ��     7               8              B�     9               :               ;               <               =               >               ?               @               A               B               C               D               E       �      NL21::battery::electricity,NL21::lost_load::electricity,NL21::demand_electricity::electricity,NL21::ccgt::electricity,NL21::hvac_transmission:NL22::electricity,NL21::pv_farm::electricity,NL21::hvac_transmission:NL23::electricity,NL21::hvac_transmission:NL13::electricity,NL21::wind_onshore::electricity,NL21::pv_rooftop::electricity,NL21::curtailment::electricity,NL21::hydrogen_to_power::electricityF       ]      NL34::wind_onshore::electricity,NL34::pv_farm::electricity,NL34::pv_rooftop::electricity,NL34::lost_load::electricity,NL34::curtailment::electricity,NL34::ccgt::electricity,NL34::demand_electricity::electricity,NL34::hvac_transmission:NL41::electricity,NL34::hydrogen_to_power::electricity,NL34::battery::electricity,NL34::wind_offshore::electricity   G       f      NL13::pv_farm::electricity,NL13::wind_onshore::electricity,NL13::ccgt::electricity,NL13::pv_rooftop::electricity,NL13::battery::electricity,NL13::curtailment::electricity,NL13::lost_load::electricity,NL13::hydrogen_to_power::electricity,NL13::hvac_transmission:NL21::electricity,NL13::hvac_transmission:NL11::electricity,NL13::demand_electricity::electricity  H       f      NL31::demand_electricity::electricity,NL31::wind_onshore::electricity,NL31::hvac_transmission:NL32::electricity,NL31::ccgt::electricity,NL31::pv_farm::electricity,NL31::lost_load::electricity,NL31::hydrogen_to_power::electricity,NL31::curtailment::electricity,NL31::battery::electricity,NL31::hvac_transmission:NL33::electricity,NL31::pv_rooftop::electricity  I       �      NL32::ccgt::electricity,NL32::hvac_transmission:NL31::electricity,NL32::wind_offshore::electricity,NL32::lost_load::electricity,NL32::pv_farm::electricity,NL32::hvac_transmission:NL23::electricity,NL32::demand_electricity::electricity,NL32::hvac_transmission:NL33::electricity,NL32::curtailment::electricity,NL32::battery::electricity,NL32::pv_rooftop::electricity,NL32::hydrogen_to_power::electricity,NL32::wind_onshore::electricity       J       �      NL41::hvac_transmission:NL42::electricity,NL41::hvac_transmission:NL22::electricity,NL41::pv_rooftop::electricity,NL41::wind_onshore::electricity,NL41::curtailment::electricity,NL41::hvac_transmission:NL34::electricity,NL41::hydrogen_to_power::electricity,NL41::hvac_transmission:NL33::electricity,NL41::ccgt::electricity,NL41::pv_farm::electricity,NL41::lost_load::electricity,NL41::battery::electricity,NL41::demand_electricity::electricity      K               L               M               N               O               P               Q               R               S               T               x^cg$�acC���� �q ~> �h���֯^�v��+V��9k�����/X�h�t�  	Z+FTREE  ����������������E                       �7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           x^�Ra@�B��r"��B�\@��Ѝ�+0512�70�����Փ��WPTRVQUS�@WH  d�TREE  ����������������                       pH                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           OHDR�                                                 �     7             �     7   8             shuffle            deflate            �H                ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    �     8   }*�OOCHK    ��     �       �     t  REFERENCE_LIST       dataset                                       dimension                                                                      �'             p@             p|�OHDR�         �       �                                 �P                  �P        8             shuffle            deflate            �`     �           ������������������������H     
   is_result              @                                P       DIMENSION_LIST                                    �P        �)�hOHDR�                                 @             �            �8             shuffle            deflate            G�                ������������������������H     
   is_result              @                                >       units                days since 2019-01-22 15:00:00;    	   calendar                     proleptic_gregorianP       DIMENSION_LIST                                    $i     �   �H��OCHK    �            �     t  REFERENCE_LIST       dataset                                       dimension                                                                      ��             (�
             ��
             �D             �3�                                                                                                                                                                                                            x^c`HЁ. ?���� Q�$�TREE  ����������������:                       �`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           GCOL                 �      NL23::demand_electricity::electricity,NL23::hvac_transmission:NL32::electricity,NL23::curtailment::electricity,NL23::hydrogen_to_power::electricity,NL23::hvac_transmission:NL21::electricity,NL23::pv_rooftop::electricity,NL23::ccgt::electricity,NL23::hvac_transmission:NL12::electricity,NL23::wind_onshore::electricity,NL23::pv_farm::electricity,NL23::lost_load::electricity,NL23::battery::electricity,NL23::wind_offshore::electricity              <      NL42::wind_onshore::electricity,NL42::lost_load::electricity,NL42::demand_electricity::electricity,NL42::hydrogen_to_power::electricity,NL42::ccgt::electricity,NL42::pv_farm::electricity,NL42::pv_rooftop::electricity,NL42::curtailment::electricity,NL42::battery::electricity,NL42::hvac_transmission:NL41::electricity           �      NL12::battery::electricity,NL12::hvac_transmission:NL11::electricity,NL12::pv_rooftop::electricity,NL12::hvac_transmission:NL23::electricity,NL12::lost_load::electricity,NL12::wind_onshore::electricity,NL12::curtailment::electricity,NL12::demand_electricity::electricity,NL12::pv_farm::electricity,NL12::wind_offshore::electricity,NL12::ccgt::electricity,NL12::hydrogen_to_power::electricity        �      NL33::ccgt::electricity,NL33::battery::electricity,NL33::hvac_transmission:NL41::electricity,NL33::wind_onshore::electricity,NL33::lost_load::electricity,NL33::hvac_transmission:NL32::electricity,NL33::wind_offshore::electricity,NL33::pv_farm::electricity,NL33::curtailment::electricity,NL33::pv_rooftop::electricity,NL33::hvac_transmission:NL31::electricity,NL33::hydrogen_to_power::electricity,NL33::demand_electricity::electricity              f      NL22::lost_load::electricity,NL22::wind_onshore::electricity,NL22::pv_farm::electricity,NL22::demand_electricity::electricity,NL22::ccgt::electricity,NL22::battery::electricity,NL22::hvac_transmission:NL41::electricity,NL22::hvac_transmission:NL21::electricity,NL22::pv_rooftop::electricity,NL22::hydrogen_to_power::electricity,NL22::curtailment::electricity         �      NL11::curtailment::electricity,NL11::hvac_transmission:NL12::electricity,NL11::lost_load::electricity,NL11::hydrogen_to_power::electricity,NL11::ccgt::electricity,NL11::pv_farm::electricity,NL11::hvac_transmission:NL13::electricity,NL11::demand_electricity::electricity,NL11::wind_offshore::electricity,NL11::wind_onshore::electricity,NL11::battery::electricity,NL11::pv_rooftop::electricity                              ��     	               
                                                                                                                                                                                                                                                                                                                                                          !               "               #               $               %               &               '               (               )               *               +               ,               -               .               /               0               1               2               3               4               5               6               7               8               9               :               ;               <               =               >               ?               @               A               B               C               D               E               F               G               H               I               J               K               L               M               N               O               P               Q               R               S               T               U               V               W               X               Y               Z               [               \               ]               ^               _               `               a               b               c               d               e               f               g               h               i               j               k               l               x^kO۸Ѧ}BlZ��]�H��-` �`������XY��]��=<��� ~�TREE  ����������������#                      $�             �                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              GCOL                                                                                                                                   	               
                                                                                                                                                                                                                                                                                                                                  %       NL21::demand_electricity::electricity                   NL32::wind_offshore::electricity!       %       NL42::demand_electricity::electricity   "              NL12::pv_rooftop::electricity   #       $       NL32::hydrogen_to_power::electricity    $              NL34::wind_onshore::electricity %       %       NL33::demand_electricity::electricity   &              NL21::pv_rooftop::electricity   '              NL13::wind_onshore::electricity (              NL13::ccgt::electricity )       )       NL23::hvac_transmission:NL21::electricity       *       %       NL12::demand_electricity::electricity   +       $       NL12::hydrogen_to_power::electricity    ,              NL34::pv_rooftop::electricity   -       )       NL23::hvac_transmission:NL12::electricity       .       )       NL33::hvac_transmission:NL31::electricity       /       )       NL34::hvac_transmission:NL41::electricity       0       $       NL23::hydrogen_to_power::electricity    1              NL21::battery::electricity      2              NL21::curtailment::electricity  3       )       NL11::hvac_transmission:NL13::electricity       4              NL13::pv_rooftop::electricity   5              NL41::wind_onshore::electricity 6              NL22::ccgt::electricity 7              NL11::lost_load::electricity    8              NL11::pv_farm::electricity      9              NL32::lost_load::electricity    :              NL33::pv_rooftop::electricity   ;       )       NL22::hvac_transmission:NL21::electricity       <              NL32::pv_rooftop::electricity   =       $       NL34::hydrogen_to_power::electricity    >              NL21::wind_onshore::electricity ?       )       NL21::hvac_transmission:NL23::electricity       @       %       NL34::demand_electricity::electricity   A              NL11::battery::electricity      B              NL33::pv_farm::electricity      C       $       NL33::hydrogen_to_power::electricity    D              NL34::lost_load::electricity    E              NL33::curtailment::electricity  F              NL42::lost_load::electricity    G              NL22::curtailment::electricity  H       )       NL22::hvac_transmission:NL41::electricity       I       )       NL32::hvac_transmission:NL31::electricity       J              NL42::pv_farm::electricity      K              NL31::curtailment::electricity  L              NL31::lost_load::electricity    M              NL42::wind_onshore::electricity N       )       NL33::hvac_transmission:NL32::electricity       O              NL11::wind_onshore::electricity P              NL21::pv_farm::electricity      Q              NL32::curtailment::electricity  R              NL11::ccgt::electricity S               NL34::wind_offshore::electricityT       %       NL31::demand_electricity::electricity   U              NL33::wind_onshore::electricity V              NL31::battery::electricity      W              NL12::lost_load::electricity    X              NL31::pv_rooftop::electricity   Y              NL34::ccgt::electricity Z              NL22::pv_rooftop::electricity   [       )       NL41::hvac_transmission:NL33::electricity       \       $       NL22::hydrogen_to_power::electricity    ]              NL11::curtailment::electricity  ^              NL11::pv_rooftop::electricity   _              NL13::battery::electricity      `       )       NL31::hvac_transmission:NL33::electricity       a       )       NL12::hvac_transmission:NL23::electricity       b              NL23::pv_farm::electricity      c              NL31::pv_farm::electricity      d              NL23::curtailment::electricity  e              NL34::pv_farm::electricity      f              NL42::ccgt::electricity g       )       NL41::hvac_transmission:NL34::electricity       h              NL12::ccgt::electricity i              NL22::pv_farm::electricity      j              NL32::ccgt::electricity k       $       NL21::hydrogen_to_power::electricity    l       )       NL32::hvac_transmission:NL33::electricity       m       )       NL32::hvac_transmission:NL23::electricity       n               NL11::wind_offshore::electricityo       )       NL33::hvac_transmission:NL41::electricity       p       %       NL41::demand_electricity::electricity   q       %       NL22::demand_electricity::electricity   r              NL13::pv_farm::electricity      s              NL33::lost_load::electricity    t              NL33::ccgt::electricity u              NL41::battery::electricity      v       $       NL13::hydrogen_to_power::electricity    w              NL42::battery::electricity      x               NL33::wind_offshore::electricityy              NL41::lost_load::electricity    z       )       NL13::hvac_transmission:NL11::electricity       {       )       NL12::hvac_transmission:NL11::electricity       |       %       NL32::demand_electricity::electricity   }              NL32::wind_onshore::electricity ~              NL12::curtailment::electricity                NL21::lost_load::electricity    �       $       NL41::hydrogen_to_power::electricity    �               NL23::wind_offshore::electricity�       )       NL41::hvac_transmission:NL22::electricity       �              NL13::curtailment::electricity  �              NL42::pv_rooftop::electricity   �       )       NL31::hvac_transmission:NL32::electricity       �              NL22::lost_load::electricity    �              NL33::battery::electricity      �       )       NL13::hvac_transmission:NL21::electricity       �              NL41::curtailment::electricity  �              NL41::ccgt::electricity �              NL12::wind_onshore::electricity �              NL42::curtailment::electricity  �               NL12::wind_offshore::electricity�              NL12::pv_farm::electricity      �              NL23::battery::electricity      �       )       NL41::hvac_transmission:NL42::electricity       �       %       NL23::demand_electricity::electricity   �              NL22::wind_onshore::electricity �       )       NL21::hvac_transmission:NL13::electricity       �              NL32::pv_farm::electricity      �              NL31::ccgt::electricity �              NL23::lost_load::electricity    �       $       NL31::hydrogen_to_power::electricity    �              NL13::lost_load::electricity    �              NL22::battery::electricity      �              NL23::ccgt::electricity �       %       NL13::demand_electricity::electricity   �       %       NL11::demand_electricity::electricity   �              NL34::battery::electricity      �              NL34::curtailment::electricity  �       $       NL42::hydrogen_to_power::electricity    �       )       NL11::hvac_transmission:NL12::electricity       �       )       NL23::hvac_transmission:NL32::electricity       �       )       NL42::hvac_transmission:NL41::electricity       �       )       NL21::hvac_transmission:NL22::electricity       �              NL31::wind_onshore::electricity �              NL41::pv_rooftop::electricity   �              NL32::battery::electricity      �              NL23::pv_rooftop::electricity   �              NL23::wind_onshore::electricity �              NL12::battery::electricity      �              NL21::ccgt::electricity �       $       NL11::hydrogen_to_power::electricity    �              NL41::pv_farm::electricity      �              Y�             8                                                                                                                                                                                                                                                                                                              x^���NEA`ހ�$�0wwwww�������C2����|��o�&ELk�
N� ���!H��L��H�&�TI����)�� �3[p4��$��#�p"�0)a�8�.�BY��R\-��s6�����������qs{w�����������pxt|rzv~qyu�����������������_X\Z^Y][�����CiO/o����
	��������OHLJNIMK��������/(,*.)-+��������ohljnimkw��fU���TREE  ����������������                       w�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           x^c`�    