?($	??n?????????????1??%???!?L?J???$	k??c??@z(??Qr@	@*?@!b?}{.-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-?????*??	??A?|a2U??YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k?????H??A6?>W[???YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??\m????"?uq??A0*??D??Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???mV}??b??Az?,C???Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c??j?q?????A????<,??YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??@????&S????A?HP???Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)?Ǻ???ŏ1w-??A?5^?I??Y?j+??ݣ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&y?&1????9#J{???A?c]?F??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l???U0*????AP??n???Y?j+??ݣ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?L?J???-C??6??A??JY?8??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
.?!??u??Tt$?????AL7?A`???Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ??7?[ A??AW[??????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г??6<?R?!??A;M?O??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_????????A'???????Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
??ˡE?????A2U0*???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR??Q?|a2??A??\m????Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_???g??s???A	?c???Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????????(??AS?!?uq??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}??????_vO??A??@?????Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O?????$??C??A?
F%u??Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1??%???{?/L?
??A~8gDi??Y???H??*	43333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?6?[ ??!޷?I$?A@)?s????1?<+??3>@:Preprocessing2F
Iterator::Model??6?[??!?MO!?!A@)?@??ǘ??16\?97@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+??ݓ???!YX?&oP@)m????ҽ?1ç?)@:Preprocessing2U
Iterator::Model::ParallelMapV2=
ףp=??!?~Dc?&@)=
ףp=??1?~Dc?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice46<???!_?2?@)46<???1_?2?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ǘ????!a???C?+@)?!??u???1cbDT?L@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*5?8EGr??!???l?i@)5?8EGr??1???l?i@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2w-!???!12???1@)?R?!?u??1??^Fb@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?hc?X_@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?§??Y?????1????*??	??!-C??6??	!       "	!       *	!       2$	??????????$??~8gDi??!??JY?8??:	!       B	!       J$	:Hu'Ŵ???^?Jj?????Q???!S?!?uq??R	!       Z$	:Hu'Ŵ???^?Jj?????Q???!S?!?uq??JCPU_ONLYY?hc?X_@b Y      Y@q???HWU@"?
both?Your program is POTENTIALLY input-bound because 48.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.0678% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 