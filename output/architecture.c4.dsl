workspace "SimpleInventoryService" "C4 model generated automatically from ADL" {
  model {
    softwareSystem SimpleInventoryService "SimpleInventoryService" {
      container ProductService "Product Service" "Handles product related operations" "API"
      container ProductRepository "Product Repository" "Stores and retrieves product data" "Service"
      container ProductController "Product Controller" "Handles incoming HTTP requests and returns responses" "Service"
      container InventoryServiceAPI "Inventory Service API" "Provides API endpoints for product operations" "API"
    }
    ProductController -> ProductService "data-flow" "HTTP"
    ProductService -> ProductRepository "data-flow" "HTTP"
    ProductController -> InventoryServiceAPI "event-flow" "Message Queue"
    InventoryServiceAPI -> ProductService "event-flow" "Message Queue"
  }
  views {
    container SimpleInventoryService {
      include *
      autoLayout lr
    }
    systemContext SimpleInventoryService {
      include *
      autoLayout tb
    }
  }
}