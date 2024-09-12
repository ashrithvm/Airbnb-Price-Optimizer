# Goals

## Business Goals
The project has three main **business goals**:
1. **Maximize yearly profit** for a specific Airbnb listing through optimization.
2. **Create a dynamic pricing tool** that adjusts daily prices to remain competitive with local listings.
3. **Design a repeatable model** that can be expanded to other locations.

## Analytics Goals
To achieve these, three **analytics goals** were established:
1. **Determine demand** in relation to price using k-means clustering of similar properties.
2. **Predict the intrinsic value** of a property using kNN regression based on its attributes.
3. **Optimize daily pricing** through a model that maximizes yearly profit by analyzing demand, property attributes, and revenue vs. costs.

# Assumptions

**Overall Model**:
- Competing listings are other Airbnbs, not hotels, assuming customers have already chosen Airbnb.
- Prices vary by month, weekday, weekend, and special events but remain constant within those periods, so the data isn’t time-sensitive.
- No partial rentals; a listing can only be rented by one party at a time.
- Only one reservation per rental period, following Airbnb’s policy.
- Sellers are assumed to already own the property and have sufficient data for the model.

**Variables**:
- Parking capacity matches the property’s capacity, so excess parking needs aren’t considered.
- Costs increase by $X per person for parties larger than two.
- Monthly upkeep includes mortgage, utilities, cleaning, HOA fees, etc., as a single cost estimate.

# Data Sources
Airbnb Listing information
http://insideairbnb.com/get-the-data.html

An additional dataset that is from the San Francisco public safety database for all reported police department incidents to model safety scores
https://data.sfgov.org/Public-Safety/Police-Department-Incidents/tmnf-yvry

# System Architecture
![alt text](https://user-images.githubusercontent.com/30711638/48392141-fce4a480-e6d7-11e8-92bf-1c542a421edd.png)

Our project focuses on determining the optimal daily listing price to maximize long-term profit. Initially, we debated between time series regression and optimization modeling. Time series regression captures daily price fluctuations and factors like property type and amenities but doesn’t guarantee an optimal profit-maximizing price. Optimization is ideal for finding the best price but becomes complex when accounting for 365 days and property characteristics. We combined both approaches by using regression in variable creation and simplifying the model with assumptions.

The system architecture separates data from two sources: a GeoJSON file for listing locations and historical data on facility details and bookings. We used clustering analysis to group similar listings within neighborhoods, identified competing properties, and performed regression analysis to create a demand function. Using kNN regression, we established a baseline price for each listing, which was incorporated into the optimization model to generate the best daily price based on booking probabilities.

# Data Model 

## Occupancy Rate

Based on Airbnb’s report for San Francisco, we assumed an average stay of 4.2 nights per booking. Since the guest review rate wasn't available, we assumed it to be 0.5. Additionally, we capped the maximum occupancy rate at 0.95 to account for occasional unrented nights. These assumptions were used to formulate the estimated occupancy rate.

![alt text](https://user-images.githubusercontent.com/30711638/48401720-c3bc2c80-e6f7-11e8-84e7-82cd1dbe25dd.png)

## Demand Function

Initially, we used separate models to predict listing price and demand. One model used k-Means clustering to group nearby similar properties, estimating monthly demand from the average occupancy rate. Another kNN regression model predicted daily listing price based on property attributes like amenities and safety. Monthly profit was estimated by multiplying the average price by predicted demand.

However, since demand and price are strongly correlated, we combined them into a single model. We prioritized modeling demand first and then incorporated it into the price function, accounting for competitor factors, listing characteristics, and time fluctuations.

![alt text](https://user-images.githubusercontent.com/30711638/48401408-1a753680-e6f7-11e8-9a04-f78caca21b7e.png)


## Competition Analysis
We model the competitor factor by using the same approach that we initially planned to apply for our monthly demand estimation. Customers tends to choose their Airbnb with a specific location in mind, so all listings that are located close together (within 3-mile radius) will be more likely to compete with each other. Going beyond this, the characteristics and quality of the listing should have an almost equal, and occasionally greater impact on determining occupancy rates relative to those properties competing with one another. From these guides we elected to use k-Means to cluster similar properties within a 3-mile radius distance (figure 2&3). 
![alt text](https://user-images.githubusercontent.com/30711638/48395835-7aafac80-e6e6-11e8-97ee-334b0695feb4.png)

## Formulate Demand Function of “Competition” Group

For each cluster, we have a set of data with the X-variable being the listing price and the Y-variable being the demand represented by the occupancy rate. From this we will fit either a  linear or polynomial regression model onto this dataset to find the best-fitted function of demand  
$D_{xij} = \alpha* x_{ij}^2 + *x_{ij} + \beta$.  
We then put this demand function into the optimization model. The objective is to maximize profit in one year, so the formula to calculate profit for each day is the listing price on that day multiplied by the demand function on that day, represented as $D_{xij} * x_{xij}$. It is worth noting that at this point in the analysis, demand in our dataset is the occupancy rate, which represents the probability a property is booked on a given day. We will mention the details of how to calculate the optimal yearly profit in the Decision Model later. 
	After several trials, we determined that the relationship between demand and price is not always inversely correlated together and straightforward linear; outlier-type cases in particular don’t fit into our model, meaning that extremely expensive listings seem to have different rules of demand governing them, which is reasonable but not accounted for in our limited model. 
![alt text](https://user-images.githubusercontent.com/30711638/48396056-2eb13780-e6e7-11e8-8c02-1a989dca7e46.png)

## Prediction of Intrinsic Property Values
We first set out to incorporate the property-specific characteristics into the model to customize the pricing model for each listing; as mentioned previously if we include too many variables into an optimization, we are less likely to have a valid solution. Instead we used our regression model to determine a “baseline” price, which we define as a price mark that indicates the intrinsic value of an Airbnb property without considering time series fluctuation (seasonalities, weekend vs weekday). Then in the optimization model we display the daily price $$x_{ij}$$  in terms of the defined baseline price with the constraint  
$$x_{ij} = \beta_{ij} * x_0$$  
The optimization model will find the optimal coefficient ij, and consequently the optimal daily listing price xij. 
	Additionally we used the k Nearest Neighbor (kNN) regression to determine the most accurate fitted baseline price for each listing, based on attributes that we decided will determine the value of a listing such as location, amenities, review scores, safety, etc. The reason we selected the kNN algorithm is because it determines the response variable Y based on the values of X-variables from k neighbors, which in our case is the values of nearby Airbnb competitors. This is a good model fit for our data because in real life, it is usually the case that real estate, hotel, and Airbnb values are heavily influenced by their surrounding competitors as a function of their location.

# Optimization Model
### Time series factors
In attempting to find the daily optimal price, we determined that our optimization model will have up to 365 price variables x1,..., x365. We originally simplified the model by assuming that price is relatively consistent throughout one month. However when further analyzing past data, we observed that Airbnb’s demand is generally higher during weekend than weekday, so we further delineated our model into two separate optimization models for both weekends and weekdays. From this we arrived at our model’s assumptions that listing price should be constant for all weekends within a month and for all weekdays within a month. 

### Decision Variables:
- ij: binary variable which is 1 if customers decide to book Airbnb property on day j of month i, equal to 0 otherwise
- $$x_{ij}$$: price of Airbnb property on day j of month i
Parameters:
- $$C_V$$ is variable cost that includes cleaning fee, utilities, guest-included fee, deterioration fee on amenities. For simplicity of our model, we will let users independently input $$C_V$$ to the application, allowing us to treat $$C_V$$ as a constant. Moreover, we assume $$C_V$$ will be incurred only on the days where bookings happened.  
- $$I_O$$ is initial investment the property owner supplies, which potentially includes the real estate or reimbursement costs, amenity purchases, maintenance cost or any other upfront cost. For simplicity of our model, we will let users input the estimate initial investment on their property, allowing us to treat this as a fixed cost throughout our model.  
- $$D_{ij}$$: demand of day j on month i, represented by the probability that the listing is actually booked on a specific day, given the listing price xij. $$D_{ij}$$ is the function of $$x_{ij}$$ and is determined by performing regression analysis on the cluster that the specific Airbnb property belongs to.

### Optimization Model

At this point we had constructed the components necessary for us to begin formulating our final optimization model. As defined above, our demand function will indicate the probability that the property is booked on day i, month j; if this “booked” probability is bigger than 50%, then we assume the property is marked as booked in our model on day i month j. This then allows the host to collect the optimal listing price, $$x_{ij}$$ and subtract the variable cost, $$C_V$$. The binary variable ij is treated as a control because the bookings on the property as a function of its probability. Therefore, the function to estimate monthly profit (months simplified to all be 30 days in length) will be:

![alt text](https://user-images.githubusercontent.com/30711638/48392310-c22f3c00-e6d8-11e8-9ec6-870bb1972dac.png)

 If occupancy of that month is 45%, then the expected number of days booked in that month is 45% * 30 days = 14 days. We can then retrieve the exact date of that month where j = 1 or D(xj) > 50%, and sum up their corresponding predicted prices to get the expected profit of that month. 

### Objective function:
The objective function is to maximize yearly profit, which equals to the revenue subtracted from the cost. The revenue is the sum of optimal listings prices on the days that bookings happen, or in the words, where ij = 1. We take the summation of both weekend and weekday (determined by index j) and of 12 months (determined by index i).

![alt text](https://user-images.githubusercontent.com/30711638/48392350-f0148080-e6d8-11e8-8853-b7197ddeed35.png)

### Output
![alt text](https://user-images.githubusercontent.com/30711638/48452409-0116cd80-e77d-11e8-9c6a-7016d58226af.png)

We tested two example listings in our optimization model:

1. **Boutique Hotel in Bayview**: Classified as a "high-end" listing based on high safety (score > 7) and luxury amenities. The kNN regression predicted an intrinsic value of $132. The model provided 24 price recommendations for weekdays and weekends over 12 months. The projected yearly profit for this listing was $25,378.

2. **Bed and Breakfast in Chinatown**: This "medium-category" listing had lower safety and amenity scores, with lower demand despite a lower price. Following the same process, the expected yearly profit for this listing was $19,764.


## UI Prototype
Figure 7 shows the first page of the application where hosts input details about their Airbnb property, such as type, number of bedrooms/bathrooms, neighborhood, and amenities. Hosts can either paste the property URL for automatic parsing or manually enter the information. They also set an estimated initial investment and variable booking costs.  
Figure 8 displays the results from the kNN regression model, predicting the property's intrinsic value, along with a map view of the property location and details.

![atl text](https://user-images.githubusercontent.com/30711638/48452192-e4c66100-e77b-11e8-847a-43a4aef0ffef.png)

Figure 9 displays the results of the optimization model, showing the optimal daily listing price on a calendar. For example, if the decision variable t7,1 = $349, then all weekdays in July are priced at $349, excluding weekends and special events. The figure includes a time series plot comparing neighborhood Airbnb demand (bottom left) and the selected listing’s demand (bottom right), offering hosts insights into their competitiveness.

Figure 10 shows the reporting tools, highlighting expected monthly demand and profit. Demand is based on occupancy rates, while profit is calculated from the predicted bookings. The highest demand and profit months are marked in gold, with a total yearly profit summary provided.

![atl text](https://user-images.githubusercontent.com/30711638/48452195-e728bb00-e77b-11e8-874e-56fb7866e197.png)

## System Dynamics
### Potential Issuses
### Issues and Monitoring Methods

- **Issue 1**: Users must fill in all listing attributes, including optional ones, as the regression model requires them. Missing attributes will cause the model to malfunction.
   - **Monitoring**: A script will check new listings for missing attributes and notify users if any fields are left empty.

- **Issue 2**: The model assumes uniform weekday and weekend pricing within a month, which may not reflect fluctuating demand throughout the week.
   - **Monitoring**: Monitor air traffic to San Francisco as an indicator of travel flow. If flight numbers fluctuate significantly (using a moving average benchmark), the system will send a notification.

### Methods for fixing/ resolving the issues 
- ### Solutions to Issues

- **Issue 1**: Since missing attributes are optional, we can auto-fill them using the most common values for categorical variables and median values for numerical ones from the Airbnb dataset. For example, if reviews are missing, we will autofill with the median review score from other listings.
  
- **Issue 2**: If a notification indicates significant demand fluctuations (based on travel data), the engineering team will analyze the trend. If notifications become frequent, we could switch from a clustering-based model to a time-series regression model to better predict daily demand based on listing characteristics.

# Conclusion
When comparing our two examples, we found that despite price differences and cluster variations, both listings had a similar demand (around 50%). This indicates a stable customer base for each market segment: some customers seek high-end, luxury Airbnbs, while others look for affordable options. Understanding where your property fits within the market is crucial for pricing. For high-end listings, rather than lowering prices to attract more customers, it’s better to invest in amenities to appeal to the right segment and avoid reducing profit margins.

## Future Work
There are several future directions for this project. Adding new variables, such as seasonality, will improve accuracy, helping predict demand and pricing more precisely on a weekly basis. Expanding to new cities will require adapting variables and gathering additional data, but the methodology remains the same. Over time, our models will improve with more historical data and insights into successes and failures.

Engaging customers will also be key. Customer feedback through reviews helps identify areas for improvement and boosts property credibility. Additionally, targeted advertising can help us reach the right audience, giving us a competitive edge.
