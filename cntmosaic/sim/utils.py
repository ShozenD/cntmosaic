from scipy.ndimage import gaussian_filter

def symmetrise_patterns(patterns: dict) -> dict:
	"""Symmetrise the contact patterns by averaging the values of the upper and lower triangles.
	
	Parameters
	----------
	patterns: dict
		Dictionary of contact patterns. Usually the output of load_contact_patterns.
  
	Returns
	-------
	dict
		Dictionary of symmetrised contact patterns.
	"""
	
	x = patterns['household']
	x[:,84] = x[:,83]
	patterns['household'] = (x + x.T) / 2

	x = patterns['community']
	x[:,84] = x[:,83]
	patterns['community'] = (x + x.T) / 2

	x = patterns['school']
	patterns['school'] = (x + x.T) / 2

	x = patterns['work']
	patterns['work'] = (x + x.T) / 2
	
	return patterns

def smooth_patterns(patterns: dict) -> dict:
	"""Smooth the contact patterns by applying a Gaussian filter to each pattern.
 
	Parameters
	----------
	patterns: dict
		Dictionary of contact patterns. Usually the output of load_contact_patterns.
  
	Returns
	-------
	dict
		Dictionary of smoothed contact patterns.
	"""
		
	patterns['household'] = gaussian_filter(patterns['household'], sigma=1)
	patterns['school'] = gaussian_filter(patterns['school'], sigma=1)
	patterns['work'] = gaussian_filter(patterns['work'], sigma=1)
	patterns['community'] = gaussian_filter(patterns['community'], sigma=1)
	
	return patterns