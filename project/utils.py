


class Statistics:
  def __init__(self):
    self.values = dict()

  def step(self, key, value):
    sum, count = 0.0, 0.0
    if key in self.values:
      sum, count = self.values[key]
    sum += value
    count += 1.0
    self.values[key] = (sum, count)

  def get(self):
    result = dict()
    for k, (sum,count) in self.values.items():
      result[k] = float(sum/count)
    return result
  
  