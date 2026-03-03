# Goal: Classify data using SVM polynomial kernel




class SupportVector:
    def __init__(self):
        pass

    def choose_parameters(self):
        pass # EU FIZ ALGO PARECIDO ANTES


    def train_test(self):
        logger.info('Vectorization text with TF-IDF')
        X = self.vectorizer.fit_transform(self.data['text'])

        logger.info("Divide train and test")
        y = self.data['label_num']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.debug(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def fitting_data(self):
        logger.info('Fitting with Bayes MultinomialNB')

        self.model = MultinomialNB(alpha=1.0)  # Smoothing

        self.model = self.model.fit(self.X_train, self.y_train)

    def predict_model(self):
        logger.info('Predict Test Data')
        self.predict_values = self.model.predict(self.X_test)

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.y_test, self.predict_values),
            display_labels=['Ham', 'Spam']
        )
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Spam Classification")
        plt.show()

        logger.debug(f'Precision Score: {precision_score(self.y_test, self.predict_values)}')
        logger.debug(f'Recall score: {recall_score(self.y_test, self.predict_values)}')

    def plot_svm(self):
        pass
