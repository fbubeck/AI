?($	F%u???1????(????y?)??!?4?8EG??$	?ce"<@???6@????@!??? ?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?!??u???M??St$??A?f??j+??YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}8???lV}???A?X?? ??Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|????e??a???A?sF????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????O??e??A??(????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG??>?٬?\??A'???????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+???гY?????A?d?`TR??Y??ͪ?զ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H.?!??S?!?uq??A????Q??Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???o??ʡ??A?0?*??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??W?2????G?z???A?JY?8???YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??HP??bX9????AF??_???Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ꕲq???o???T???A8gDio??Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ???(??0??A6?;Nё??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'???M?J???A:#J{?/??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??MbX??lxz?,C??A?????B??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C???ͪ??V??A?!??u???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??x?&1??E???JY??A?1w-!??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ڊ?e??}??b???A&S????YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W?/?'????z6???AKY?8????Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
??(??y??A	??g????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?X????.???1???Au?V??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)???A?fշ?A??	h"l??Y???H??*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatԚ?????!s?.??F@)z6?>W??17????E@:Preprocessing2F
Iterator::Model?&?W??!?9v???<@)d;?O????1?q???1@:Preprocessing2U
Iterator::Model::ParallelMapV2:#J{?/??!,?'K%/&@):#J{?/??1,?'K%/&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd;?O????!?q???Q@)???JY???1?#D??M#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceT㥛? ??!$??vKc@)T㥛? ??1$??vKc@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate`vOj??!?^g??&'@)&S????1*?R.?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}?5^?I??!ݜ???/@)46<?R??1j|\???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*J+???! ,??@)J+???1 ,??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o:?m?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?JY?8????L?+???M??St$??!>?٬?\??	!       "	!       *	!       2$	?r?/????h??*?????	h"l??!'???????:	!       B	!       J$	E?LEԛ???<?b??2U0*???!Ǻ????R	!       Z$	E?LEԛ???<?b??2U0*???!Ǻ????JCPU_ONLYYo:?m?@b Y      Y@qJ>????>@"?
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.8776% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 