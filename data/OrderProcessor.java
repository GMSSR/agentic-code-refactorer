public class OrderProcessor {
    private String orderId;
    private double amount;
    private String customerName;
    private String customerEmail;
    private String street;
    private String city;
    private String zipCode;

    public OrderProcessor(String orderId, double amount, String customerName, String customerEmail, String street, String city, String zipCode) {
        this.orderId = orderId;
        this.amount = amount;
        this.customerName = customerName;
        this.customerEmail = customerEmail;
        this.street = street;
        this.city = city;
        this.zipCode = zipCode;
    }

    public void processOrder() {
        System.out.println("Processing order: " + orderId + " for " + customerName);
        if (amount > 1000) {
            sendHighValueAlertEmail();
        }
        sendConfirmationEmail();
    }

    private void sendHighValueAlertEmail() {
        System.out.println("Sending high value alert to " + customerEmail);
    }

    private void sendConfirmationEmail() {
        System.out.println("Sending confirmation email to " + customerEmail);
    }
}
