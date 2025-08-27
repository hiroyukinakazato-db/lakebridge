-- Sample T-SQL for E2E testing
-- This is a minimal test file to verify Switch conversion functionality

SELECT 
    c.customer_id,
    c.customer_name,
    o.order_date,
    o.total_amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
ORDER BY o.order_date DESC;

-- Test stored procedure conversion
CREATE PROCEDURE UpdateCustomerStatus
    @CustomerId INT,
    @NewStatus NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    
    UPDATE customers 
    SET status = @NewStatus 
    WHERE customer_id = @CustomerId;
    
    SELECT 'Customer status updated successfully' AS Result;
END;