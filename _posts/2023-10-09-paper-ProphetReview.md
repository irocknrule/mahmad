---
title: "Paper review: Forecasting at Scale"
date: 2023-10-09T15:34:30-04:00
categories:
  - Blog
  - Forecasting
tags:
  - Paper Reviews
  - Time Series
  - Forecasting
classes: wide
---

Have not posted in a while, so I wanted to review and write about some time-series forecasting algorithms that I have used previously in my work since they have been coming up again recently. Time series forecasting is incredibly interesting but equally hard to do right as a whole lot of domain knowledge is generally needed to actually evaluate if a forecast is good or not. In my time in AWS Network capacity, hardware and traffic forecasting was an area of emphasis with wide ramifications across the business. In this post, I will review the Facebook Prophet time series forecasting paper which brings an interesting angle to the *forecasting at scale* problem.

## Background

Time Series forecasts often involve forecasting based on thousands of data points for many different applications or systems. For example, in data centers we could want to forecast network traffic growth between racks communicating with each other, services talking within a specific rack, overall traffic between data centers or the rate of growth of various components within the data centers. Different markets, geographies and locations will have very different growth patterns and millions of time series forecasts could be needed at various levels within the network to serve in planning purposes. 

The paper 'Forecasting at Scale' by Taylor and Letham from Facebook approach this problem of generating huge numbers of forecasts from a different perspective of scale - they do not focus on typical scale considerations such as compute power and storage. Scale for them invokes how we can invoke a 'human-in-the-loop' with expert (or adequate) domain knowledge to help configure the forecasts easily and then let automation handle the hard task of model evaluation. This is a really interesting approach towards generating large amounts of forecasts for capacity planning or anomaly detection and in my opinion an effective proposal towards a good solution to this hard problem. Figure 1 presents a view of this approach which best makes use of both a human and an automated system.

{% include figure image_path="/assets/images/blogs/fb_prophet_fig1.png" alt="" caption="Figure 1: Proposed forecasting approach with automation and analyst-in-the-loop."%}{: .align-left style="width: 35%;"}



The authors use Figure 2 as their demo dataset, which comprises of events created on Facebook for every day of the week for the years between 2013 to 2017. As can be seen from the figure, the times series includes multiple weekly and yearly seasonalities, trends and outliers. Also holiday effects are very evident too. 

{% include figure image_path="/assets/images/blogs/fb_prophet_fig2.png" alt="" caption="Figure 2: Dataset used in this paper. Contains number of events created on Facebook from 2013 to 2017 with each day of the week being color coded."%}{: .align-left style="width: 75%;"}


## Prophet Model

The authors propose a decomposable time series model with 3 components: **trend**, **seasonality** and **holidays**. They are combined in the following simple equation:

{% include figure image_path="/assets/images/blogs/fb_prophet_eq1.png" alt="" caption="Equation 1: The decomposable time series model with the main components."%}{: .align-left style="width: 35%;"}

where

g(t) is: **Trend** - a nonlinear saturating growth model or piecewise linear model with automatic changepoint selection. It allows modeling growth trends and trend changes.

s(t) is - **Seasonality:** Modeled additively. Can accommodate multiple seasonalities like weekly and yearly cycles.

h(t) is - **Holidays:** Irregular events like holidays are explicitly modeled.

e_t is - an error term representing idiosyncratic changes not accommodated by the model.

Essentially, the forecasting problem is framed as a curve-fitting exercise which provides flexibility to accomodate multiple seasonalities or irregular events and intuitive parameters for non-experts to adjust the forecast parameters based on their domain knowledge. It is also easily extensible with new components and fits very fast. 

As an aside, it is important to understand what is a saturating growth model - growth slows and eventually levels off as it approaches a saturation point or carrying capacity. It captures exponential growth that slows and approaches a limit, which is useful for modeling many types of constrained growth processes. The logistic curve is a classic example. Some key characteristics of a saturating growth model are:

- Growth is initially exponential but slows as it approaches a limit or ceiling. The rate of growth decreases over time.
- There is a saturation point or carrying capacity where growth levels off. This is the maximum population size that can be supported within limited resources.
- Common saturating growth models include the logistic growth model and variations like the generalized logistic curve. These model growth with an S-shaped curve.
- Saturating growth models are commonly used to model constrained growth situations like population growth, market adoption of a new product, or growth of a company/platform.
- Key advantages are the ability to model an initial exponential growth phase and a slowdown and saturation. This matches many real-world growth trajectories.

Changepoint selection can be carried out automatically once the analyst specifies specific known dates which may affect the trends or other growth-altering events. In the author's use case, they provide table 1 as the example list of holidays which affect the overall event creation numbers on facebook. 

{% include figure image_path="/assets/images/blogs/fb_prophet_tab1.png" alt="" caption="Table 1: Example list of holidays provided to the model."%}{: .align-left style="width: 35%;"}

## Model fitting results

{% include figure image_path="/assets/images/blogs/fb_prophet_fig3.png" alt="" caption="Figure 3: Prophet forecast for the entire time period with interpolation into the future as the forecast. We can observe the model learns about the relevant seasonalities and that the overall trend at the end of 2016 is generally increasing."%}{: .align-left style="width: 75%;"}

Figure 3 above presents the results of using Prophet on the input dataset with the solid lines representing the curve-fitting while the dashed line represents the out-of-sample forecast. We can see that individual end of year seasonalities have been captured by Prophet whereas the increasing trend at the end of 2016 going in to 2017 are also showing up in the forecasts. Other traditional forecasting techniques missed these (at the time of this paper being published).
## Comparison with other modeling techniques

An overall comparison of the different time series forecasting techniques are provided below: 

|Method|Trend Model|Seasonality Model|Holiday Model|Changepoints|Parameters|
|---|---|---|---|---|---|
|Prophet (Proposed)|Piecewise linear or logistic growth|Fourier series decomposition|Flexible specification|Automatically detected or manually specified|Intuitive parameters like growth rate, seasonality period, holiday magnitude|
|ARIMA|Linear trend|Seasonal AR terms|None|None|Maximum AR, MA, seasonal AR, seasonal MA orders|
|Exponential Smoothing|Linear trend|Multiplicative seasonality|None|None|Smoothing parameters for level, trend, seasonality|
|TBATS|Piecewise linear trend|Fourier series decomposition|None|Automatically detected|ARMA components, Fourier series terms, error distribution|

## Recent trends in time series forecasting

There has been some recent work in comparing Prophet to other forecasting approaches and its been observed that Prophet generally does not work well when compared to more recent LSTM based approaches, ARIMA or even conventional CNNs. It does not generalize well to all types of forecasting approaches, which in my opinion is both a strength and a weakness. If you can find a particular use case which Prophet works well with, then there is no harm to stick to it. What Prophet does well is it identifies seasonality and enables the analyst-in-the-loop to provide direction. There is some interesting new work in PyTorch with the release of *NeuralProphet* \[2] which I have not used yet but plan on implementing and writing about soon. Some more info about the future of Prophet was also published earlier this at \[3]. The authors state that they will mostly be in maintenance mode and no new changes/enhancements will be made to the underlying modeling algorithms.



# References

\[1] Forecasting at Scale: https://facebook.github.io/prophet/

\[2] NeuralProphet: Explainable Forecasting at Scale: https://neuralprophet.com/ 

\[3] facebook/prophet in 2023 and beyond: https://medium.com/@cuongduong_35162/facebook-prophet-in-2023-and-beyond-c5086151c138
