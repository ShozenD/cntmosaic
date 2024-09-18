# If socialmixr is not installed, install it
install.packages("socialmixr", repos = "http://cran.us.r-project.org")
library(socialmixr)

df <- polymod$contacts

# Save the data to a csv file
write.csv(df, "data/polymod/contacts.csv", row.names = FALSE)