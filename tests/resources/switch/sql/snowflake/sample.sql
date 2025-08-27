-- Sample Snowflake SQL for E2E testing
-- This is a minimal test file to verify Switch conversion functionality

SELECT 
    customer_id,
    customer_name,
    order_date,
    total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE order_date >= '2024-01-01'
ORDER BY order_date DESC;

-- Test stored procedure conversion
CREATE OR REPLACE PROCEDURE update_customer_status(customer_id INT, new_status STRING)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    UPDATE customers SET status = :new_status WHERE id = :customer_id;
    RETURN 'Customer status updated successfully';
END;
$$;