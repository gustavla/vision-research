# This is a Google Sketchup plugin for generated images for various angles

require 'sketchup.rb'

Sketchup.send_action "showRubyPanel:"

def gaussian(mean, stddev)
    if stddev == 0.0 then
        return mean
    end
    theta = 2 * Math::PI * rand()
    rho = Math.sqrt(-2 * Math.log(1 - rand()))
    scale = stddev * rho
    x = mean + scale * Math.cos(theta)
    y = mean + scale * Math.sin(theta)
    return [mean+stddev*2, [mean-stddev*2, x].max].min
end



UI.menu("Plugins").add_item("Generate bicycles") {

    #prompts = ['Name of this model?']
    #defaults = ['carN']
    #input = UI.inputbox prompts, defaults, 'Give us some info'

    # Set up rendering style 
    styles = Sketchup.active_model.styles
    srand(0)
    
    puts styles["gustav"].class
    style = styles["gustav"]
    if style.instance_of?(Sketchup::Style) then
        styles.selected_style = style
    else
        styles.add_style "/Users/slimgee/git/vision-research/cad/sketchup/gustav.style", true
    end

    model = Sketchup.active_model
    model.entities.each do |entity|
        puts entity 
        if entity.is_a? Sketchup::Group
            puts 'This is a group'
            puts entity.name
            entity.visible = false 
        end
    end

    N = 40 
    SAMPLES_PER_VIEW = 1 
    SD = 0.0

    model.entities.each do |entity|
        if entity.is_a? Sketchup::Group
            name = entity.name
            entity.visible = true
            
            view = model.active_view
            puts('view', view)
            si = 256 

            save_image = Proc.new { |viewname, i, altitude, azimuth, out_of_plane, target, dist0, focal_length|
                dist = dist0 * focal_length / 67.0
                #rotation = outofplane_index * 6.0 * Math::PI / 180.0
                x = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.cos(azimuth)
                y = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.sin(azimuth) 
                z = dist * Math.cos(Math::PI / 2.0 - altitude) 

                eye = [x + target[0], y + target[1], z + target[2]]
                up= [
                    Math.sin(out_of_plane) * Math.cos(azimuth + Math::PI / 2.0), 
                    Math.sin(out_of_plane) * Math.sin(azimuth + Math::PI / 2.0), 
                    Math.cos(out_of_plane)
                ]
                #filename = "/Users/slimgee/git/data/xi3zao3-car-sides7/view%03d_%s.png" % [i, name]
                filename = "/Users/slimgee/git/data/sketchup-output/%s%03d_%s.png" % [viewname, i, name]
                if x != 0 or y != 0 then
                    if not File.exists? filename then
                        camera = Sketchup::Camera.new eye, target, up
                        #puts 'perspective', camera.perspective?
                        camera.perspective = true
                        camera.focal_length = focal_length
                        #puts 'Camera focal length'
                        #puts camera.focal_length
                        view.camera = camera
                        keys = {
                            :filename => filename,
                            :width => si,
                            :height => si,
                            :antialias => true,
                            :compression => 0.9,
                            :transparent => true,
                        }
                        status = view.write_image keys 
                    end
                    true
                else
                    false
                end 
            } 

            #dist = 140 # Bike

            range = (0..N).map { |i| -1 + 2.0 * i/N.to_f } 
            # step(0.1) does not work in SketchUp's Ruby version
            dist = 500 
            dist2 = 330  # car
            dist3 = 450
            i = 0

            if false 
                (0..2).each do |altitude_index|
                    altitude = ((rand() - 0.5) * 4 + altitude_index * 15.0) * Math::PI / 180.0
                    # Do only half
                    (0...18).each do |azimuth_index|
                        #if .. skip front/back
                        #[0, -1, 1].each do |outofplane_index|
                            #outofplane_index = rand(3) - 1


                            # Also change the azimuth angle a bit with the outofplane
                            azimuth = (rand() * 4 + azimuth_index * 20.0) * Math::PI / 180.0

                            if save_image.call(i, altitude, azimuth, 0.0, [0, 0, 0], dist, 67.0)
                                i += 1
                            end
                        #end
                    end
                end 
            end

            if false 
               #(0...18).each do |azimuth_index|
               (0...72).each do |azimuth_index|
                    azimuth = (rand() * 4 + azimuth_index * 20.0/4.0) * Math::PI / 180.0
                    altitude = (rand() * 7.5) * Math::PI / 180.0
                    out_of_plane = (rand() - 0.5) * 2.5 * Math::PI / 180.0
                    focal_length = 65 + (rand() - 0.5) * 15
                    if save_image.call(i, altitude, azimuth, out_of_plane, [0.0, 0, 0], dist, focal_length)
                        i += 1
                    end
                end
            end

            # half-side cars
            if true 
                #[[25, 3.5], [50, 25]].each do |angle, altitude0|
                [[25, 3.5], [50, 13.5]].each_with_index do |(angle, altitude0), ai|
                    2.times do |flip1|
                        2.times do |flip2|
                            SAMPLES_PER_VIEW.times do |loop|
                                #angle = 15
                                #[40, 65, 80].each do |focal_length|
                                [65].each do |focal_length|
                                    focal_length += gaussian(0, 8 * SD)
                                    rand_altitude = gaussian(0, 3 * SD)
                                    altitude = (altitude0 + rand_altitude) * Math::PI / 180.0
                                    #azimuth = (-45 + 90 * flip + (rand() - 0.5) * 7.5) * Math::PI / 180.0      #sides1
                                    #azimuth = (-45 + 90 * flip + (rand() - 0.5) * 20.0) * Math::PI / 180.0     #sides2
                                    rand_azimuth = gaussian(0, 10 * SD)
                                    puts('flip1', flip1)
                                    puts('flip2', flip2)
                                    puts('altidue', altitude0)
                                    puts('ai', ai)
                                    puts('angle', angle)
                                    puts('rand_azimuth', rand_azimuth)
                                    azimuth = (90 + angle * (2*flip1-1) + 180 * flip2 + rand_azimuth) * Math::PI / 180.0
                                    puts('AFTER')
                                    vid = 2 + flip1 * 2 + ai 
                                    rand_out_of_plane = gaussian(0, 2.5 * SD)
                                    out_of_plane = rand_out_of_plane * Math::PI / 180.0
                                    if save_image.call("#{vid}-half-side", i, altitude, azimuth, out_of_plane, [0.0, 0, 0], dist3, focal_length)
                                        i += 1
                                    end
                                end
                            end
                        end
                    end
                end
            end

            # Profiles
            if true 
                2.times do |flip|
                    SAMPLES_PER_VIEW.times do |loop|
                        [65].each do |focal_length|
                            focal_length += gaussian(0, 15 * SD)
                            altitude = rand() * 7.5 * Math::PI / 180.0
                            rand_azimuth = gaussian(0, 10 * SD)
                            azimuth = (180 * flip + rand_azimuth) * Math::PI / 180.0
                            rand_out_of_plane = gaussian(0, 2.5 * SD)
                            out_of_plane = rand_out_of_plane * Math::PI / 180.0
                            if save_image.call('0-profile', i, altitude, azimuth, out_of_plane, [2.5, 0, 0], dist, focal_length)
                                i += 1
                            end
                        end
                    end
                end
            end

            # Save some specialized frontal shots
            if true
                2.times do |flip|
                    sgn = [1, -1][flip]
                    SAMPLES_PER_VIEW.times do |loop|
                        [65].each do |focal_length|
                            focal_length += gaussian(0, 15 * SD)
                            #if save_image.call(i, rand() * 15 * Math::PI / 180.0, (180.0 * flip + 90.0 + (rand() - 0.5) * 15) * Math::PI / 180.0, (rand() - 0.5) * 5 * Math::PI / 180.0, [0, sgn * 80, 0], dist2)
                            rand_out_of_plane = gaussian(0, 2.5 * SD)
                            rand_azimuth = gaussian(0, 10 * SD)
                            if save_image.call('1-front', i, SD * rand() * 10 * Math::PI / 180.0, (180.0 * flip + 90.0 + rand_azimuth) * Math::PI / 180.0, rand_out_of_plane * Math::PI / 180.0, [0, 0, 0], dist2, focal_length)
                            #if save_image.call(i, rand() * 20 * Math::PI / 180.0, (180.0 * flip + 90.0 + (rand() - 0.5) * 20) * Math::PI / 180.0, (rand() - 0.5) * 10 * Math::PI / 180.0, [0, sgn * 80, 0], dist2)
                            #if save_image.call(i, rand() * 30 * Math::PI / 180.0, (180.0 * flip + 90.0 + (rand() - 0.5) * 30) * Math::PI / 180.0, (rand() - 0.5) * 10 * Math::PI / 180.0, [0, sgn * 80, 0], dist2)
                                i += 1
                            end
                        end
                    end
                end
            end
            entity.visible = false 
        end
    end
    UI.messagebox("Data saved!")
}

