# Constants
resource_area_open_field_pv = 0.0125  # km²/MW
resource_area_wind_onshore_competing = 0.125  # km²/MW

# Function to calculate the energy cap based on available area and resource area per energy capacity
def calculate_energy_cap(available_area, resource_area_per_energy_cap):
    return available_area / resource_area_per_energy_cap

# Given data
available_area = 0.0 * 10000  # Convert from 10,000 km² to km²

# Split the available area (50% for open field PV and 50% for wind onshore competing)
area_open_field_pv = available_area * 0.5
area_wind_onshore_competing = available_area * 0.5

# Calculate energy caps
energy_cap_open_field_pv = calculate_energy_cap(area_open_field_pv, resource_area_open_field_pv)
energy_cap_wind_onshore_competing = calculate_energy_cap(area_wind_onshore_competing, resource_area_wind_onshore_competing)