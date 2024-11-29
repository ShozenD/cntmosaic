import matplotlib.pyplot as plt

def plot_base_patterns(patterns: dict, **kwargs):
	fig, ax = plt.subplots(1, 4, **kwargs)
	
	for i, item in enumerate(patterns.items()):
		name, pattern = item
		ax[i].imshow(pattern, cmap='Spectral_r', origin='lower')
		
		ax[i].set_title(name.capitalize(), fontsize=9, loc='Left')
		ax[i].tick_params(axis='both', which='major', labelsize=8)
		ax[i].tick_params(axis='both', which='minor', labelsize=8)
		
		ax[i].set_xlabel('Age of contacting', fontsize=8)
		
		if i == 0:
			ax[i].set_ylabel('Age of contacted', fontsize=8)
		else:
			ax[i].set_ylabel('')
			ax[i].set_yticklabels([])
	
	fig.tight_layout()
	plt.show()