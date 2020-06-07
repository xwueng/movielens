# -------- movielens machine learning script --------
# Functionality of this script:
# 1. Download movielens file, merge movies.dat and rating.dat and create 
# the training dataset, edx, and validation data set
# 2. Preprocess edx and validation by extracting movie release year from title and 
# 3. Analyze user, movie title and genres stats 
# 4. Train prediction models and compute RMSEs 
# 5. Plot predictor values

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")

# -------- File download ---------
# Download movielens file, merge movies.dat and rating.dat and create 
# the training dataset, edx, and validation data set
# Note: this process could take a couple of minutes
# movielens project splits movielens data into a 
# training set edx and a validation set. 

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
# ----- edx is the train set
edx <- movielens[-test_index,] 
temp <- movielens[test_index,]

# validation is the test set
# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)



rm(dl, ratings, movies, test_index, temp, movielens, removed)

# --------Preprocess data sets ----------
preprocess_movies <- function(movies) {
  # convert timestamp to rating date: rtdate
  # extract movie release year from title and add rlyear column
  # add movie age: mvage = rating year - movie release year
  year_pattern <- "\\(\\d{4}\\)$"
  rlyear <- str_match(movies$title, year_pattern)
  rlyear <- as.numeric(str_remove_all(rlyear,  "\\(|\\)"))
  movies <- movies %>% mutate(rlyear=rlyear,
                              rtdate = round_date(as_datetime(timestamp), unit="week"),
                              mvage = year(rtdate) - rlyear) %>% 
    select(-timestamp)
  return(movies)
}

edx <- preprocess_movies(edx)
mu <- mean(edx$rating)
validation <- preprocess_movies(validation)

# --------Prepare edx and validation tables for project report ----------
edx_after <- edx[1:2,] %>% knitr::kable(align="r", caption="New edx data")

edx_summary_tbl <- edx %>% summarize(n_users = n_distinct(userId), 
                                     n_movies=n_distinct(movieId), 
                                     n_genres = n_distinct(genres),
                                     n_ratings = n()) %>% 
  setnames(c("User Count", "Movie Title Count", "Genres Count", "Rating Count")) %>% 
  knitr::kable(caption = "edx Data Set Summary", align="c")
edx_summary_tbl

validation_summary_tbl <- validation %>% summarize(n_users = n_distinct(userId), 
                                                   n_movies=n_distinct(movieId), 
                                                   n_genres = n_distinct(genres),
                                                   n_ratings = n()) %>% 
  setnames(c("User Count", "Movie Title Count", "Genres Count", "Rating Count")) %>% 
  knitr::kable(caption = "validation Data Set Summary", , align="c")

validation_summary_tbl

# ----- Analyze user, movie title and genres stats --------
options(digits = 3)

# ----show edx's rating counts and quantile
edx$rating %>% table() %>% t() %>%  kable()


# -----User Analysis ---
# user rating distribution (quantile)

# average nrating/user, movies/user, generes/user
user_stats_edx <- edx %>% group_by(userId) %>% 
  summarize(n_ratings=n(), 
            rating_avg=mean(rating), 
            rtyears=max(year(rtdate)) -  min(year(rtdate))) 
nratings_avg <-mean(user_stats_edx$n_ratings)
rating_avg <- mean(edx$rating)
rtyears_avg <- mean((user_stats_edx$rtyears))

user_stats_tbl <- tibble(nratings_avg, rating_avg, rtyears_avg) %>% 
  setnames(c("Average Number of Ratings per User", "Rating Average", "Average Rating Experience (years)")) %>% 
  knitr::kable(caption = "User Statistics", , align="c")     

user_stats_tbl

p_user_stats_edx <- user_stats_edx %>% 
  ggplot(aes(n_ratings, rating_avg, color=rtyears)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Rating Average and Rating Count by User", 
       x = "Rating Count by User",
       y = "Average Rating by User") + 
  geom_vline(xintercept = nratings_avg, show.legend=TRUE) + 
  geom_hline(yintercept = rating_avg, show.legend=TRUE) + 
  annotate("text", x = 1500, 
           y = 0.75, label = paste0("count avg: ", format(nratings_avg, digits =3))) +
  annotate("text", x = 5000, 
           y = rating_avg+0.25, label = paste0("rating avg: ", format(rating_avg, digits =3))) +
  # geom_text(aes(0, rating_avg ,label = rating_avg, color="red")) +
  scale_color_gradientn(colours = rainbow(5, rev=TRUE)) 

p_user_stats_edx
# ggsave("figs/user_stats_edx.png", plot=p_user_stats_edx)

# -----Movie Analysis ---
# rating distribution (histogram)
# titles/genre
# movies/genres correlation
title_stats_edx <- edx %>% 
  group_by(movieId, title) %>% 
  summarize(n_ratings=n(), rating_avg=mean(rating), mvage_avg=mean(mvage)) 

p_title_stats_edx <- title_stats_edx %>%  
  ggplot(aes(n_ratings, rating_avg, color=mvage_avg)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Rating Average and Rating Count by Movie Titles", 
       x = "Rating Count by Title",
       y = "Average Rating by Title") + 
  scale_fill_discrete(name="Movie Age") +
  scale_color_gradientn(colours = rainbow(5, rev=TRUE)) 

p_title_stats_edx
# ggsave("figs/title_stats_edx.png", plot=p_title_stats_edx)


# ----Genres Analysis --
# rating quaquantile(by_genere)
genres_stats_edx <- edx %>% group_by(genres) %>%
  summarize(n_ratings=n(), rating_avg=mean(rating)) 


#-- Temporal Factor Analysis----
# nrating vs. year
# rating avgs vs. year
# movie age vs. rating

rlyear_stats_edx <- edx %>% group_by(rlyear) %>% 
  summarize(n_ratings=n(), rating_avg=mean(rating), mvage_avg=mean(year(rtdate) - rlyear)) 

p_rlyear <-rlyear_stats_edx %>%  ggplot(aes(rlyear, rating_avg, color=n_ratings)) +
  geom_point() +
  geom_smooth() +
  scale_color_gradientn(colours = rainbow(5, rev=TRUE)) 

p_rlyear
# ggsave("figs/rlyear_stats_edx.png", p_rlyear)

# -------Rating correlations -----
rating_quantile <- quantile(edx$rating) %>% 
  t() %>%
  knitr::kable(caption = "Rating Quantile", , align="c")
rating_quantile

predictors <- c("Title", "User", "Genres", "Movie Release Year", "Movie Age")
cors <- c(cor(as.numeric(as.factor(edx$title)), edx$rating),
          cor(edx$userId, edx$rating), 
          cor(as.numeric(as.factor(edx$genres)), edx$rating),
          cor(edx$rlyear, edx$rating),
          cor(edx$mvage, edx$rating)
)
rating_correlations <- tibble(predictor = predictors, rating_cor = cors)
rating_correlations 


# ------ Split edx to train_set and test_set ---
# test_set set will be 10% of edx data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
# ----- train set
train_set <- edx[-test_index,] 
temp <- edx[test_index,]

# test set:
# Make sure userId and movieId in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(temp, removed)

# ---Train data model and collect RMSEs ----

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
} 
# ----compute rmses ----------
# ----Training Function ----------
compute_rmses <- function(lambda, train_set, test_set, include_biases = 3, return_artifact=FALSE) {
  # Description
  # compute rating bias from predictors:
     # b_i: movie title "i", 
     # b_u: user "u",
     # b_g: genres "g" 
     # b_t: movie age "t" = movie review year - release year
  # Arguments 
     # lambda: tuning parameter 
     # train_set: training data set
     # test_set: test data set
  # include_biases: controls which predictors to include:
    #   1: no predictor, just mu
    #   2: b_i and b_u
    #   3: b_i,  b_u, b_t, b_g
  # return_artifact: controls whether to return biases in result which 
    # can be used to plot bias values and help visualize predictor effects 
    #   return_artifact = TRUE: return result, result is test_set with biases
   #   return_artifact = FALSE: return RMSE
  
  options(digits = 5)
  mu <- mean(train_set$rating)
  
  if (include_biases == 1) {
    # include_biases == 0: don't include biases in prediction
    # predicted rating = mu 
    result <- test_set %>%
      mutate(pred = mu)
  }
  
  # if include_biases = 2 or 3 continue to compute b_i and b_g
  if (include_biases %in% c(2, 3)) {
    # b_i: movie item bias
    b_i <- train_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+lambda))
    
    # b_u: user bias
    b_u <- train_set %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
    
    # save result for baseline 2 only
    if (include_biases == 2) {
      result <- test_set %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(pred = mu + b_i + b_u)
    }
  }
  
  # if include_biases = 3 continue to compute b_t and b_g
  if (include_biases == 3) { 
    # b_t: movie age bias
    b_t <- train_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      group_by(mvage) %>%
      summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+lambda))
    
    # b_g: genres bias
    b_g <- train_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_t, by = "mvage") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - b_i - b_u - b_t - mu)/(n()+lambda))
    
    # compute predicted ratings for the test_set and 
    # save the biases and pred in result 
    result <- test_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_t, by = "mvage") %>%
      left_join(b_g, by = "genres") %>%
      mutate(pred = mu + b_i + b_u + b_t + b_g)
  } #end baseline 3
  
  predicted_ratings <-
    result %>% pull(pred)
  
  rmse <- RMSE(predicted_ratings, test_set$rating)
  
  # control return content based on return_artifact
  ifelse (return_artifact,
            return(result),  #return biases in result
            return(rmse))    #just return RMSE
  
}


options(digits = 5)

# ------ Baseline 1: Use train_set and test_set to compute RMSE for mu ---
bl1_rmses <- sapply(0, compute_rmses, train_set=train_set, 
                    test_set= test_set, include_biases=1, return_artifact=FALSE)
bl1 <- c(method="Baseline 1: predicted_rating = mu", RMSE=format(bl1_rmses, nsmall = 5), best_lambda= NA)
rmse_results <- bl1
rmse_results


# ------ Baseline 2: Use train_set and test_set to compute RMSE for b_i and b_u ---
bl2_lab <- "Baseline 2: predicted_rating = mu + b_i + b_u"
lambdas <- seq(4, 7, 0.25)
bl2_rmses <- sapply(lambdas, compute_rmses, train_set=train_set, 
                    test_set= test_set,  include_biases = 2, return_artifact = FALSE)

# ---- check and display baseline 2 results
print(bl2_min_rmse <- min(bl2_rmses))
print(bl2_best_lambda <- lambdas[which.min(bl2_rmses)])
# rbind(lambdas, bl2_rmses) %>% saveRDS(., "rdas/bl2_lambdas_rmses.rda")

bl2 <- c(method=bl2_lab, 
         RMSE=format(bl2_min_rmse, nsmall=5), 
         best_lambda=bl2_best_lambda)

print(rmse_results <- rbind(rmse_results, bl2))


# ------ Baseline 3: Use train_set and test_set to compute RMSE for b_i, b_u, b_g, b_t ---
bl3_lab <- "Baseline 3: predicted_rating = mu + b_i + b_u + b_t + b_g"
lambdas <- seq(4, 7, 0.25)
bl3_rmses <- sapply(lambdas, compute_rmses,train_set=train_set, 
                    test_set= test_set, include_biases = 3, return_artifact = FALSE)
bl3_rmses

# -- Select best lambda: bl3_best_lambda
print(bl3_min_rmse <- min(bl3_rmses))
print(bl3_best_lambda <- lambdas[which.min(bl3_rmses)])
# rbind(lambdas, bl3_rmses) %>% saveRDS(., "rdas/bl3_lambdas_rmses.rda")

print(bl3 <- c(method=bl3_lab, 
               RMSE=format(bl3_min_rmse, nsmall=5), 
               best_lambda=bl3_best_lambda))

p__bl3_lambdas <- qplot(lambdas, bl3_rmses, xlab = bl3_lab)
p__bl3_lambdas

rmse_results

# +++++ compute FINAL RMSE using edx and validation +++
# bl3_best_lambda is from baseline 3 
final_lab <- "Final: predicted_rating = mu + b_i + b_u + b_t + b_g"
final_rmse <- compute_rmses(bl3_best_lambda, train_set=edx, 
                            test_set=validation, include_biases=3, return_artifact=FALSE)
final_rmse
print(final <- c(method=final_lab, 
               RMSE=format(final_rmse, nsmall=5), 
               best_lambda=bl3_best_lambda))

rmse_results <- rbind(bl1, bl2, bl3, final) %>% 
  kable(align="c", caption = "RMSE for mu + b_i + b_u + b_t + b_g")
# saveRDS(final_rmse, "figs/final_rmse.rda")

# --- Get result which is validation with b_*
# ----result will be used to create plots to visualize bias effect ---
result <- compute_rmses(bl3_best_lambda, train_set=edx, 
                        test_set=validation, include_biases=3, return_artifact=TRUE)

# --- Calculate biases' quantiles
bias_quqntiles <- rbind(c("b_i", format(quantile(result$b_i), digits =3)),
                        c("b_u", format(quantile(result$b_u), digits =3)),
                        c("b_g", format(quantile(result$b_g), digits =3)),
                        c("b_t", format(quantile(result$b_t), digits =3))
                       ) %>% 
  kable(caption = "Predictors Quantiles")


# --------- Create Bias specific plots --------------
# take 500,000 random samples from result and use them to make plots

result <- result[sample(1:nrow(result), nrow(result)/2, replace = FALSE),] %>% 
  melt(id.vars = c("rating", "pred"), measure.vars = c("b_i","b_u", "b_t", "b_g"))


# stacked line plots to show rating adjustments made by b_i and b_u
# x: predicted rating
# (plotting takes 5 mins)

p_pred_bias_iu_line <- result %>% 
  filter((variable %in% c("b_i", "b_u"))) %>% 
  ggplot(aes(x=pred, y=value, fill=variable, color=variable)) + 
  geom_line() + 
  labs(title = "Movie Item b_i and User b_u Quantiles", 
       subtitle = "(true ratings labeled on boxes)", 
       y = "bias value") +
  theme(plot.title = element_text(hjust = 0.5)) +
  facet_wrap(~rating)

p_pred_bias_iu_line
# ggsave("figs/pred_bias_iu_line.png", p_pred_bias_iu_line)
