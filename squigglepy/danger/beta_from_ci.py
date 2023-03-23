import requests

def get_beta_from_ci(ci_lower, ci_upper, ci_length):
	url = 'https://trastos.nunosempere.com/fit-beta'
	data = {
		"ci_lower": ci_lower,
		"ci_upper": ci_upper,
		"ci_length": ci_length # actually optional
	}
	response = requests.post(url, json = data)
	json_response = response.json()
	return [json_response[0], json_response[1]]

answer = get_beta_from_ci(0.1, 0.8, 0.9)
print(answer)
