# Create class and set initial DSR threshold
class LoanCalculator:
    def __init__(self):
        self.loans = []
        self.dsr_threshold = 70

# Get user input
    def get_user_input(self):
        principal = self.get_valid_input("Enter principal loan amount (RM): ", float)
        annual_interest_rate = self.get_valid_input("Enter the annual interest rate (%): ", float)
        loan_term_years = self.get_valid_input("Enter the loan term (years): ", int)
        monthly_income = self.get_valid_input("Enter monthly income (RM): ", float)
        num_commitments = self.get_valid_input("Enter the number of other monthly financial commitments: ", int)

# Get the amount of monthly commmitments
        commitments = []
        for i in range(num_commitments):
            commitment_amount = self.get_valid_input(f"Enter the amount for commitment (RM) {i + 1}: ", float)
            commitments.append(commitment_amount)

        return principal, annual_interest_rate, loan_term_years, monthly_income, commitments

# Validate input
    def get_valid_input(self, prompt, data_type):
        while True:
            try:
                user_input = data_type(input(prompt))
                if user_input < 0:
                    print("Please enter a non-negative value.")
                else:
                    return user_input
            except ValueError:
                print("Invalid input. Please enter a valid numerical value.")

# Define function to calculate monthly_instalment
    def calculate_monthly_instalment(self, principal, annual_interest_rate, loan_term_years):
        monthly_interest_rate = (annual_interest_rate / 100) / 12
        total_payments = loan_term_years * 12
        monthly_instalment = (principal * monthly_interest_rate) / (1 - (1 + monthly_interest_rate) ** -total_payments)
        return monthly_instalment

# Define function to calculate total amount payable
    def calculate_total_amount_payable(self, monthly_instalment, loan_term_years):
        return monthly_instalment * loan_term_years * 12

# Define function to calculate Debt Service Ratio (DSR)
    def calculate_dsr(self, monthly_income, commitments, monthly_instalment):
        total_monthly_commitments = sum(commitments) + monthly_instalment
        dsr = (total_monthly_commitments / monthly_income) * 100
        return dsr

# Display loan details
    def display_loan_details(self, principal, monthly_instalment, total_amount_payable, dsr):
        print("\nLoan Details:")
        print(f"Principal Loan Amount: RM{principal:.2f}")
        print(f"Monthly Instalment: RM{monthly_instalment:.2f}")
        print(f"Total Amount Payable: RM{total_amount_payable:.2f}")
        print(f"Debt Service Ratio (DSR): {dsr:.2f}%")
        print("Eligible for the loan: " + ("Yes" if dsr <= self.dsr_threshold else "No"))

# Calculate loan
    def calculate_loan(self):
        principal, annual_interest_rate, loan_term_years, monthly_income, commitments = self.get_user_input()
        monthly_instalment = self.calculate_monthly_instalment(principal, annual_interest_rate, loan_term_years)
        total_amount_payable = self.calculate_total_amount_payable(monthly_instalment, loan_term_years)
        dsr = self.calculate_dsr(monthly_income, commitments, monthly_instalment)

        self.display_loan_details(principal, monthly_instalment, total_amount_payable, dsr)

        self.loans.append({
            "Principal": principal,
            "Monthly Instalment": monthly_instalment,
            "Total Amount Payable": total_amount_payable,
            "DSR": dsr
        })

# Display previous loan
    def display_all_loans(self):
        print("\nPrevious Loan Calculations:")
        for i, loan in enumerate(self.loans, 1):
            print(f"\nLoan {i}:")
            for key, value in loan.items():
                print(f"{key}: RM{value:.2f}")

# Modify Debt Service Ratio (DSR)
    def modify_dsr_threshold(self):
        new_threshold = self.get_valid_input("Enter the new DSR threshold: ", float)
        self.dsr_threshold = new_threshold
        print(f"DSR threshold has been updated to {new_threshold}%")

# Delete selected previous loan
    def delete_loan(self):
        self.display_all_loans()
        loan_to_delete = self.get_valid_input("Enter the number of the loan to delete: ", int)

        if 1 <= loan_to_delete <= len(self.loans):
            deleted_loan = self.loans.pop(loan_to_delete - 1)
            print(f"Loan {loan_to_delete} has been deleted.")
            print(f"Deleted Loan Details:")
            for key, value in deleted_loan.items():
                print(f"{key}: ${value:.2f}")
        else:
            print("Invalid loan number.")

# Main program loop
    def main_menu(self):
        while True:
            print("\nMain Menu:")
            print("1. Calculate a new loan")
            print("2. Display all previous loan calculations")
            print("3. Modify DSR threshold")
            print("4. Delete a previous loan")
            print("5. Exit")
            choice = self.get_valid_input("Enter your choice (1/2/3/4/5): ", int)

            if choice == 1:
                self.calculate_loan()
            elif choice == 2:
                self.display_all_loans()
            elif choice == 3:
                self.modify_dsr_threshold()
            elif choice == 4:
                self.delete_loan()
            elif choice == 5:
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

#End of class
if __name__ == "__main__":
    loan_calculator = LoanCalculator()
    loan_calculator.main_menu()
