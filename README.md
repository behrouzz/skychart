**Author:** [Behrouz Safari](https://behrouzz.github.io/)<br/>
**Website:** [AstroDataScience.Net](https://astrodatascience.net/)<br/>

# Sky Chart
Creating star charts with python

## Example 1: 

Let's create sky chart of Paris at this moment. We want just the stars with apparent magnitude below 5.

```python
import skychart as sch
from datetime import datetime
import matplotlib.pyplot as plt

t = datetime.now()
obs_loc = (2.3522, 48.8566)

fig, ax, df = sch.draw(obs_loc, t, mag_max=5, alpha=0.3)
plt.show()
```


## Example 2: 

Here we make the same chart, using the low-level function *draw_chart* :

```python
import skychart as sch
from datetime import datetime
import matplotlib.pyplot as plt

t = datetime.now()
obs_loc = (2.3522, 48.8566)

# Base dataframe
df = sch.visible_hipparcos(obs_loc, t)

# DataFrame of stars that will be shown
df_show = df[df['Vmag']<5]

# Load constellation data
dc_const = sch.load_constellations()

# Show only Ursa Major and Cassiopeia constellations
dc_const = {'UMa': dc_const['UMa'],
            'Cas': dc_const['Cas']}

fig, ax, df_show = sch.draw_chart(df, df_show, dc_const, alpha=0.3)
plt.show()
```