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

{% include figure image_path="/assets/images/blogs/fb_prophet_fig1.png" alt="" caption="Proposed forecasting approach with automation and analyst-in-the-loop."%}
