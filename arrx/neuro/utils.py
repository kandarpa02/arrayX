class ParamDict(dict):
    def flatten(self):
        """Return the list of numpy arrays in the correct order for Function"""
        return tuple([self[v] for v in self.keys()])
    
    def variables(self):
        return list(self.keys())