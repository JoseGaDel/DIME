
#
# Main component of TF Micro project 'micro_speech'.
#
idf_component_register(
    SRCS main.cc main_functions.cc output_handler.cc model.cc
    PRIV_REQUIRES spi_flash esp_driver_uart #esp_driver_gpio
    INCLUDE_DIRS "."
    REQUIRES esp_timer
)  # Add esp_driver_uart to PRIV_REQUIRES)
